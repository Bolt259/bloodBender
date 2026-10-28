"""
bloodTwin Dataset and DataLoader

Handles loading chronologically split LSTM data, creating sliding windows,
and preparing batches for training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import RobustScaler
import pickle
import logging

logger = logging.getLogger(__name__)

# Features that must not be shifted or rescaled.
# sin_time/cos_time: already in [-1,1]; rescaling distorts circadian signal.
# mask_bg/mask_label: binary flags; subtracting their median inverts meaning.
_PASSTHROUGH_FEATURES = frozenset(["sin_time", "cos_time", "mask_bg", "mask_label"])


class BloodGlucoseDataset(Dataset):
    """
    PyTorch Dataset for blood glucose time series prediction.
    
    Creates sliding windows of (lookback, horizon) pairs from continuous
    time series data. Handles multiple pump datasets.
    """
    
    def __init__(
        self,
        data_dir: Path,
        subject_dirs: Dict[str, Path],
        subset: str,  # 'train', 'validate', or 'test'
        features: List[str],
        target: str,
        lookback: int,
        horizon: int,
        stride: int = 1,
        scaler: Optional[RobustScaler] = None,
        fit_scaler: bool = False
    ):
        """
        Args:
            data_dir: Base directory containing lstm_pump_data
            pump_ids: List of pump serial numbers to load
            subset: 'train', 'validate', or 'test'
            features: List of feature column names
            target: Target column name (e.g., 'bg')
            lookback: Number of historical timesteps
            horizon: Number of future timesteps to predict
            stride: Step size for sliding window
            scaler: Pre-fitted scaler (for val/test)
            fit_scaler: If True, fit new scaler on this data (train only)
        """
        self.data_dir = Path(data_dir)
        self.subject_dirs = subject_dirs
        self.subset = subset
        self.features = features
        self.target = target
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.scaler = scaler
        
        # Load and concatenate data from all pumps
        self.data, self.timestamps = self._load_data()
        
        # Fit or use provided scaler
        if fit_scaler:
            self.scaler = RobustScaler()
            self.scaler.fit(self.data[self.features].values)
            # Binary flags and bounded circular encodings must not be shifted or
            # scaled — pass them through unchanged by zeroing their center and
            # fixing their scale to 1.  The formula (x-0)/1 = x is a no-op.
            # sin_time/cos_time are already in [-1,1] by construction; rescaling
            # them adds dataset-specific distortion and can suppress circadian
            # patterns (dawn, meal-time, sleep IS) the model needs to learn.
            # mask_bg/mask_label are binary flags: RobustScaler with IQR=0 falls
            # back to scale=1 but still subtracts the median, inverting/shifting
            # their meaning (mask_label median=1 → True becomes 0, False → -1).
            for i, feat in enumerate(self.features):
                if feat in _PASSTHROUGH_FEATURES:
                    self.scaler.center_[i] = 0.0
                    self.scaler.scale_[i]  = 1.0
            logger.info(f"Fitted RobustScaler on {subset} data (passthrough: {_PASSTHROUGH_FEATURES})")
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(
            f"Created {len(self.sequences)} sequences from {subset} set "
            f"({len(self.data)} total records, {len(subject_dirs)} pumps)"
        )
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        all_data = []

        for subject_id, subject_root in self.subject_dirs.items():
            split_dir = subject_root / self.subset

            csv_files = list(split_dir.glob(f"lstm_{self.subset}_*.csv"))

            if not csv_files:
                logger.warning(f"No CSV files found in {split_dir}")
                continue

            csv_file = sorted(csv_files)[-1]
            df = pd.read_csv(csv_file, comment="#")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["subject_id"] = subject_id

            all_data.append(df)

        if not all_data:
            raise ValueError(f"No data loaded for {self.subset} set!")

        combined = pd.concat(all_data, ignore_index=True)
        combined.sort_values(["subject_id", "timestamp"], inplace=True)
        combined.reset_index(drop=True, inplace=True)

        return combined, pd.DatetimeIndex(combined["timestamp"])
    
    def _create_sequences(self) -> List[Tuple[int, int]]:
        """
        Create sliding window sequence indices, filtering out sequences with missing BG values.
        
        Returns:
            List of (start_idx, end_idx) tuples for each valid sequence
        """
        sequences = []
        total_length = self.lookback + self.horizon

        for _, group in self.data.groupby("subject_id", sort=False):
            indices = group.index.to_list()

            for offset in range(0, len(indices) - total_length + 1, self.stride):
                window_indices = indices[offset:offset + total_length]
                sequence = self.data.loc[window_indices]

                if sequence[self.target].isna().any():
                    continue

                nan_ratio = (
                    sequence[self.features].isna().sum().sum()
                    / (len(sequence) * len(self.features))
                )

                if nan_ratio > 0.1:
                    continue

                sequences.append((window_indices[0], window_indices[-1] + 1))

        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.
        
        Returns:
            dict with keys:
                - 'input': (lookback, n_features) tensor
                - 'target': (horizon,) tensor (BG values)
                - 'timestamp': start timestamp (for debugging)
        """
        start_idx, end_idx = self.sequences[idx]
        
        # Extract sequence
        sequence = self.data.iloc[start_idx:end_idx]
        
        # Split into input and target
        input_sequence = sequence.iloc[:self.lookback]
        target_sequence = sequence.iloc[self.lookback:]
        
        # Get feature values
        input_features = input_sequence[self.features].values.astype(np.float32)
        
        # Fill any remaining NaNs with 0 (after filtering most NaN sequences)
        input_features = np.nan_to_num(input_features, nan=0.0)
        
        # Scale features
        if self.scaler is not None:
            input_features = self.scaler.transform(input_features)
        
        # Target BG for the horizon (no NaNs remain after _create_sequences filtering)
        target_values = target_sequence[self.target].values.astype(np.float32)

        # Observation mask: 1.0 where the horizon BG is genuinely measured, 0.0 where
        # the resampler interpolated it (mask_bg=True). Loss/metrics reduce over
        # observed steps only so the model is neither trained nor scored on imputation.
        if 'mask_bg' in target_sequence.columns:
            observed = ~target_sequence['mask_bg'].to_numpy().astype(bool)
        else:
            observed = np.ones(len(target_sequence), dtype=bool)
        target_mask = (observed & target_sequence[self.target].notna().to_numpy()).astype(np.float32)

        # Meal-rise label (indirect supervision): 1 if BG rises ≥35 mg/dL over the
        # horizon AND current BG is ≥80.  Covers post-meal highs and rebound
        # hyperglycaemia after a prolonged low.  Computed on raw (unscaled) BG.
        current_bg = input_sequence[self.target].values[-1]  # last raw BG in lookback
        valid_future = target_values[target_mask > 0]        # observed horizon BG only
        if len(valid_future) > 0 and not np.isnan(current_bg):
            max_rise = float(np.nanmax(valid_future) - current_bg)
            meal_label = float(max_rise >= 35.0 and current_bg >= 80.0)
        else:
            meal_label = 0.0

        # Per-step correction mask: 1.0 = this horizon step is pre-correction
        # (safe to include in loss), 0.0 = a patient bolus occurred at or before
        # this step (actual BG reflects the intervention, not a 0U trajectory).
        #
        # Key design decision (per Opus agent review 2026-10-27):
        #   - Mask FROM the correction step onward, NOT the whole window.
        #     Early horizon steps before the correction are still valid signal.
        #   - Applied as an ADDITIONAL mask on top of target_mask so the loss
        #     sees: observed AND pre-correction steps only.
        #   - At N=1 (single patient) this matters: 33.8% of windows have a
        #     correction in the 60-min horizon; 20.9% of horizon steps are
        #     affected.  Without this mask the 0U trajectory is systematically
        #     biased low (model learned "BG comes back down" because the patient
        #     always corrected — the intervention is absorbed into the target).
        if 'bolus_dose' in target_sequence.columns:
            horizon_boluses = target_sequence['bolus_dose'].fillna(0.0).to_numpy()
            has_correction  = horizon_boluses > 0.5
            first_corr_idx  = int(np.argmax(has_correction)) if has_correction.any() else len(has_correction)
            # 1.0 for steps before first correction, 0.0 from correction onward
            correction_step_mask = np.ones(len(target_values), dtype=np.float32)
            if has_correction.any():
                correction_step_mask[first_corr_idx:] = 0.0
        else:
            correction_step_mask = np.ones(len(target_values), dtype=np.float32)

        # Combined effective mask: observed AND pre-correction
        effective_mask = (target_mask * correction_step_mask).astype(np.float32)

        return {
            'input':                torch.from_numpy(input_features),
            'target':               torch.from_numpy(target_values),
            'target_mask':          torch.from_numpy(effective_mask),   # combined mask used for loss
            'target_mask_obs_only': torch.from_numpy(target_mask),      # observation-only (for metrics)
            'meal_label':           torch.tensor(meal_label, dtype=torch.float32),
            'corrected_in_horizon': torch.tensor(float(has_correction.any() if 'bolus_dose' in target_sequence.columns else False), dtype=torch.float32),
            'timestamp':            str(self.timestamps[start_idx]),
        }
    
    def save_scaler(self, path: Path):
        """Save the fitted scaler as a dict carrying the feature-name order.

        Persisting a bare RobustScaler loses which columns (and in what order)
        it was fit on, forcing consumers to guess feature names — the source of
        past 8-vs-11-feature mismatches. We save a dict with the scaler plus its
        feature_names/center_/scale_ so inference can align features exactly.
        The loader below unwraps it back to a RobustScaler for training reuse.
        """
        if self.scaler is None:
            raise ValueError("No scaler to save!")

        payload = {
            'scaler': self.scaler,
            'feature_names': list(self.features),
            'center_': self.scaler.center_,
            'scale_': self.scaler.scale_,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        logger.info(f"Saved scaler ({len(self.features)} features) to {path}")

    @staticmethod
    def load_scaler(path: Path) -> RobustScaler:
        """Load a saved scaler, unwrapping the dict format (with backward
        compatibility for older bare-RobustScaler pickles)."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        scaler = obj['scaler'] if isinstance(obj, dict) else obj
        logger.info(f"Loaded scaler from {path}")
        return scaler

def discover_dataset_subject_dirs(
    data_dir: Path,
    dataset_dirs: List[str],
    subset_names=("train", "validate", "test"),
    ) -> Dict[str, Path]:
    
    discovered = {}

    for dataset_name in dataset_dirs:
        dataset_root = Path(data_dir) / dataset_name

        if not dataset_root.exists():
            logger.warning(f"Dataset directory not found: {dataset_root}")
            continue

        for candidate in dataset_root.rglob("*"):
            if not candidate.is_dir():
                continue

            has_all_splits = all(
                (candidate / subset).exists()
                and list((candidate / subset).glob(f"lstm_{subset}_*.csv"))
                for subset in subset_names
            )

            if not has_all_splits:
                continue

            subject_key = str(candidate.relative_to(data_dir)).replace("/", "__")
            discovered[subject_key] = candidate

    return discovered

def create_dataloaders(
    data_dir: Path,
    pump_ids: Optional[List[str]],
    dataset_dirs: Optional[List[str]],
    features: List[str],
    target: str,
    lookback: int,
    horizon: int,
    stride: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    scaler_path: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, RobustScaler]:
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        (train_loader, val_loader, test_loader, scaler)
    """

    subject_dirs = {}

    # Manual pump folders
    for pump_id in pump_ids or []:
        pump_root = Path(data_dir) / f"pump_{pump_id}"

        if pump_root.exists():
            subject_dirs[f"pump_{pump_id}"] = pump_root
        else:
            logger.warning(f"Pump directory not found: {pump_root}")

    # Auto-discovered dataset subjects
    subject_dirs.update(
        discover_dataset_subject_dirs(
            data_dir=Path(data_dir),
            dataset_dirs=dataset_dirs or [],
        )
    )

    if not subject_dirs:
        raise ValueError(f"No pump or dataset subject folders found in {data_dir}")

    logger.info(f"Using {subject_dirs.keys()} total subject/pump directories")

    # Create training dataset and fit scaler
    train_dataset = BloodGlucoseDataset(
        data_dir=data_dir,
        subject_dirs=subject_dirs,
        subset='train',
        features=features,
        target=target,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        fit_scaler=True
    )
    
    scaler = train_dataset.scaler
    
    # Save scaler if path provided
    if scaler_path:
        train_dataset.save_scaler(scaler_path)
    
    # Create validation dataset with fitted scaler
    val_dataset = BloodGlucoseDataset(
        data_dir=data_dir,
        subject_dirs=subject_dirs,
        subset='validate',
        features=features,
        target=target,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        scaler=scaler
    )
    
    # Create test dataset with fitted scaler
    test_dataset = BloodGlucoseDataset(
        data_dir=data_dir,
        subject_dirs=subject_dirs,
        subset='test',
        features=features,
        target=target,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        scaler=scaler
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0
    )
    
    return train_loader, val_loader, test_loader, scaler
