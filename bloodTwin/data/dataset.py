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


class BloodGlucoseDataset(Dataset):
    """
    PyTorch Dataset for blood glucose time series prediction.
    
    Creates sliding windows of (lookback, horizon) pairs from continuous
    time series data. Handles multiple pump datasets.
    """
    
    def __init__(
        self,
        data_dir: Path,
        pump_ids: List[str],
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
        self.pump_ids = pump_ids
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
            logger.info(f"Fitted RobustScaler on {subset} data")
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(
            f"Created {len(self.sequences)} sequences from {subset} set "
            f"({len(self.data)} total records, {len(pump_ids)} pumps)"
        )
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Load and concatenate CSV files from all pumps."""
        all_data = []
        
        for pump_id in self.pump_ids:
            pump_dir = self.data_dir / f"pump_{pump_id}" / self.subset
            
            if not pump_dir.exists():
                logger.warning(f"Directory not found: {pump_dir}")
                continue
            
            # Find the lstm_*.csv file (not summary JSONs)
            csv_files = list(pump_dir.glob(f"lstm_{self.subset}_*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found in {pump_dir}")
                continue
            
            # Use most recent file
            csv_file = sorted(csv_files)[-1]
            logger.info(f"Loading {csv_file}")
            
            # Load CSV (skip comment lines)
            df = pd.read_csv(csv_file, comment='#')
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['pump_id'] = pump_id
            
            all_data.append(df)
        
        if not all_data:
            raise ValueError(f"No data loaded for {self.subset} set!")
        
        # Concatenate all pump data and sort by timestamp
        combined = pd.concat(all_data, ignore_index=True)
        combined.sort_values('timestamp', inplace=True)
        combined.reset_index(drop=True, inplace=True)
        
        timestamps = pd.DatetimeIndex(combined['timestamp'])
        
        return combined, timestamps
    
    def _create_sequences(self) -> List[Tuple[int, int]]:
        """
        Create sliding window sequence indices, filtering out sequences with missing BG values.
        
        Returns:
            List of (start_idx, end_idx) tuples for each valid sequence
        """
        sequences = []
        total_length = self.lookback + self.horizon
        
        # Slide window with stride
        for i in range(0, len(self.data) - total_length + 1, self.stride):
            # Check if this sequence has missing BG values
            sequence = self.data.iloc[i:i + total_length]
            
            # Skip sequences with any NaN in BG (target variable)
            if sequence[self.target].isna().any():
                continue
            
            # Also skip if too many NaNs in other features (>10%)
            nan_ratio = sequence[self.features].isna().sum().sum() / (len(sequence) * len(self.features))
            if nan_ratio > 0.1:
                continue
            
            sequences.append((i, i + total_length))
        
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
        
        # Get target (BG values for horizon) - should have no NaNs due to filtering
        target_values = target_sequence[self.target].values.astype(np.float32)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_features)
        target_tensor = torch.from_numpy(target_values)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'timestamp': str(self.timestamps[start_idx])  # Convert to string for collation
        }
    
    def save_scaler(self, path: Path):
        """Save the fitted scaler."""
        if self.scaler is None:
            raise ValueError("No scaler to save!")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler to {path}")
    
    @staticmethod
    def load_scaler(path: Path) -> RobustScaler:
        """Load a saved scaler."""
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {path}")
        return scaler


def create_dataloaders(
    data_dir: Path,
    pump_ids: List[str],
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
    # Create training dataset and fit scaler
    train_dataset = BloodGlucoseDataset(
        data_dir=data_dir,
        pump_ids=pump_ids,
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
        pump_ids=pump_ids,
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
        pump_ids=pump_ids,
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
