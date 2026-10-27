"""
bloodTwin LSTM Model

PyTorch Lightning implementation of LSTM for blood glucose prediction.
Designed for multi-horizon forecasting with physiological constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import torchmetrics
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)


class BloodTwinLSTM(pl.LightningModule):
    """
    LSTM-based blood glucose prediction model.
    
    Architecture:
    - Multi-layer LSTM encoder for sequential processing
    - Dropout for regularization
    - Linear decoder for multi-step prediction
    - Support for mixed precision training (AMP)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 12,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_params: Optional[Dict] = None,
        bg_min: float = 20.0,
        bg_max: float = 600.0,
        meal_loss_weight: float = 0.1,
        meal_pos_weight: float = 3.0,
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            horizon: Number of future steps to predict
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            scheduler_params: ReduceLROnPlateau parameters
            bg_min: Minimum physiological BG (mg/dL)
            bg_max: Maximum physiological BG (mg/dL)
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params or {}
        self.bg_min = bg_min
        self.bg_max = bg_max
        self.meal_loss_weight = meal_loss_weight

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # BG prediction decoder: final hidden → horizon BG values
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

        # Auxiliary meal-rise classification head: shared hidden → P(significant BG rise)
        # Trained with indirect supervision: label=1 when BG rises ≥threshold over horizon.
        # Small head to regularise rather than dominate; meal_loss_weight keeps it secondary.
        self.meal_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        # BCEWithLogitsLoss with pos_weight to counter class imbalance (~20-30% positive)
        self.meal_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(meal_pos_weight)
        )

        # Loss function
        self.loss_fn = nn.L1Loss()  # MAE loss (more robust to outliers)
        
        # Metrics
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        
        # Horizon-specific metrics (e.g., 30min, 60min)
        self.horizon_checkpoints = [6, 12]  # 30min and 60min @ 5-min intervals
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared LSTM encoder and return the final hidden state."""
        _, (hidden, _) = self.lstm(x)
        return self.dropout(hidden[-1])  # (batch, hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both BG predictions and meal-rise probability.

        Args:
            x: (batch_size, lookback, input_size)

        Returns:
            bg_predictions : (batch_size, horizon)  — scaled BG forecast
            meal_logit     : (batch_size,)           — raw logit for meal-rise event
                             Apply sigmoid for P(meal_rise).
        """
        h = self._encode(x)
        bg_preds = self.decoder(h)
        if not self.training:
            bg_preds = torch.clamp(bg_preds, self.bg_min, self.bg_max)
        meal_logit = self.meal_head(h).squeeze(-1)  # (batch,)
        return bg_preds, meal_logit
    
    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of `values` over positions where mask==1 (safe when no positions are set)."""
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked MAE: reduce absolute error over observed BG steps only (mask==1)."""
        return self._masked_mean(torch.abs(predictions - targets), mask)
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Forward + masked multi-task loss/metrics. `stage` in {train,val,test}.

        BG loss: masked MAE over observed horizon steps only.
        Meal loss: BCEWithLogitsLoss on the auxiliary rise-event head.
        total = bg_loss + meal_loss_weight * meal_loss
        """
        bg_preds, meal_logit = self(batch['input'])
        targets, mask = batch['target'], batch['target_mask']

        bg_loss   = self._compute_loss(bg_preds, targets, mask)
        meal_loss = torch.tensor(0.0, device=bg_loss.device)

        if 'meal_label' in batch:
            ml = batch['meal_label'].float()
            # Only compute meal loss on windows that have at least one observed target
            has_obs = (mask.sum(dim=-1) > 0)
            if has_obs.any():
                meal_loss = self.meal_loss_fn(meal_logit[has_obs], ml[has_obs])

        loss = bg_loss + self.meal_loss_weight * meal_loss

        mae, rmse = getattr(self, f'{stage}_mae'), getattr(self, f'{stage}_rmse')
        observed = mask > 0.5
        if observed.any():
            mae(bg_preds[observed], targets[observed])
            rmse(bg_preds[observed], targets[observed])

        if stage != 'train':
            for k in self.horizon_checkpoints:
                if k <= self.horizon:
                    mae_k = self._masked_mean(
                        torch.abs(bg_preds[:, :k] - targets[:, :k]), mask[:, :k]
                    )
                    self.log(f'{stage}_mae_{k*5}min', mae_k)

        prog = stage != 'test'
        self.log(f'{stage}_loss',      loss,      on_step=(stage == 'train'), on_epoch=True, prog_bar=prog)
        self.log(f'{stage}_bg_loss',   bg_loss,   on_step=False, on_epoch=True)
        self.log(f'{stage}_meal_loss', meal_loss, on_step=False, on_epoch=True)
        self.log(f'{stage}_mae',  mae,  on_step=False, on_epoch=True, prog_bar=prog)
        self.log(f'{stage}_rmse', rmse, on_step=False, on_epoch=True, prog_bar=(stage == 'val'))

        # Log meal-event accuracy on val/test
        if stage != 'train' and 'meal_label' in batch:
            meal_acc = ((meal_logit > 0).float() == batch['meal_label'].float()).float().mean()
            self.log(f'{stage}_meal_acc', meal_acc, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'test')
    
    def predict_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        inputs = batch['input']
        bg_preds, meal_logit = self(inputs)
        return {
            'predictions':   bg_preds,
            'meal_prob':     torch.sigmoid(meal_logit),
            'timestamps':    batch.get('timestamp', None),
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Only return scheduler if configured
        if self.scheduler_params is None:
            return optimizer
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.scheduler_params.get('mode', 'min'),
            factor=self.scheduler_params.get('factor', 0.5),
            patience=self.scheduler_params.get('patience', 3),
            min_lr=self.scheduler_params.get('min_lr', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mae',
                'interval': 'epoch',
                'frequency': 1
            }
        }


class BloodTwinEnsemble(nn.Module):
    """
    Ensemble of multiple LSTM models for improved predictions.
    Can be used for uncertainty quantification.
    """
    
    def __init__(self, models: List[BloodTwinLSTM]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Returns:
            mean_pred: (batch_size, horizon) - mean prediction
            std_pred: (batch_size, horizon) - standard deviation
        """
        predictions = torch.stack([model(x) for model in self.models])
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
