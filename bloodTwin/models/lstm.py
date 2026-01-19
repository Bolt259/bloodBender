"""
bloodTwin LSTM Model

PyTorch Lightning implementation of LSTM for blood glucose prediction.
Designed for multi-horizon forecasting with physiological constraints.
"""

import torch
import torch.nn as nn
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
        bg_max: float = 600.0
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
        
        # Output decoder: map final hidden state to horizon predictions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, lookback, input_size)
            
        Returns:
            predictions: (batch_size, horizon)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use final hidden state
        final_hidden = hidden[-1]  # (batch_size, hidden_size)
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        # Decode to horizon predictions
        predictions = self.decoder(final_hidden)  # (batch_size, horizon)
        
        # Apply physiological constraints (soft clipping during training)
        if not self.training:
            predictions = torch.clamp(predictions, self.bg_min, self.bg_max)
        
        return predictions
    
    def _compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with optional horizon weighting."""
        return self.loss_fn(predictions, targets)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        inputs = batch['input']
        targets = batch['target']
        
        # Forward pass
        predictions = self(inputs)
        
        # Compute loss
        loss = self._compute_loss(predictions, targets)
        
        # Update metrics
        self.train_mae(predictions, targets)
        self.train_rmse(predictions, targets)
        
        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        inputs = batch['input']
        targets = batch['target']
        
        # Forward pass
        predictions = self(inputs)
        
        # Compute loss
        loss = self._compute_loss(predictions, targets)
        
        # Update metrics
        self.val_mae(predictions, targets)
        self.val_rmse(predictions, targets)
        
        # Horizon-specific metrics
        for checkpoint in self.horizon_checkpoints:
            if checkpoint <= self.horizon:
                checkpoint_mae = torch.mean(torch.abs(
                    predictions[:, :checkpoint] - targets[:, :checkpoint]
                ))
                self.log(f'val_mae_{checkpoint*5}min', checkpoint_mae)
        
        # Log
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rmse', self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        inputs = batch['input']
        targets = batch['target']
        
        # Forward pass
        predictions = self(inputs)
        
        # Compute loss
        loss = self._compute_loss(predictions, targets)
        
        # Update metrics
        self.test_mae(predictions, targets)
        self.test_rmse(predictions, targets)
        
        # Horizon-specific metrics
        for checkpoint in self.horizon_checkpoints:
            if checkpoint <= self.horizon:
                checkpoint_mae = torch.mean(torch.abs(
                    predictions[:, :checkpoint] - targets[:, :checkpoint]
                ))
                self.log(f'test_mae_{checkpoint*5}min', checkpoint_mae)
        
        # Log
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_mae', self.test_mae, on_step=False, on_epoch=True)
        self.log('test_rmse', self.test_rmse, on_step=False, on_epoch=True)
        
        return loss
    
    def predict_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        inputs = batch['input']
        predictions = self(inputs)
        
        return {
            'predictions': predictions,
            'timestamps': batch.get('timestamp', None)
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
