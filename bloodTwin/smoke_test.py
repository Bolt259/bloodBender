"""
Quick smoke test for bloodTwin training pipeline.
Trains for 1 epoch with small batch to verify GPU and model work correctly.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bloodTwin.models.lstm import BloodTwinLSTM
from bloodTwin.data.dataset import create_dataloaders
from bloodTwin import ARTIFACTS_DIR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def smoke_test():
    """Run a quick smoke test."""
    print("="*70)
    print("bloodTwin LSTM Smoke Test")
    print("="*70)
    
    # Setup
    pl.seed_everything(42)
    
    data_dir = PROJECT_ROOT / "bloodBath/bloodBank/lstm_pump_data"
    artifacts_dir = ARTIFACTS_DIR / "smoke_test"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Create small dataloaders (larger stride for speed)
    print("\n[1/5] Creating dataloaders...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_dir=data_dir,
        pump_ids=["881235"],  # Just one pump for speed
        features=['bg', 'delta_bg', 'basal_rate', 'bolus_dose', 'sin_time', 'cos_time', 'bg_clip_flag', 'bg_missing_flag'],
        target='bg',
        lookback=288,
        horizon=12,
        stride=50,  # Large stride = fewer sequences
        batch_size=32,  # Small batch
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        scaler_path=artifacts_dir / 'scaler.pkl'
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\n[2/5] Creating model...")
    model = BloodTwinLSTM(
        input_size=8,
        hidden_size=64,  # Smaller for speed
        num_layers=2,
        dropout=0.2,
        horizon=12,
        learning_rate=1e-3,
        weight_decay=1e-5,
        scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 3},
        bg_min=20.0,
        bg_max=600.0
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Check GPU
    print("\n[3/5] Checking GPU...")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ No GPU detected, using CPU")
    
    # Create trainer
    print("\n[4/5] Creating trainer...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=artifacts_dir / 'checkpoints',
        filename='smoke-test-{epoch:02d}',
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )
    
    trainer = pl.Trainer(
        max_epochs=2,  # Just 2 epochs for smoke test
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else '32',
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10
    )
    
    # Train
    print("\n[5/5] Training (2 epochs)...")
    print("="*70)
    trainer.fit(model, train_loader, val_loader)
    
    # Quick test
    print("\n" + "="*70)
    print("Testing...")
    test_results = trainer.test(model, test_loader, ckpt_path='best')
    
    print("\n" + "="*70)
    print("🎉 Smoke Test Complete!")
    print("="*70)
    print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Test MAE: {test_results[0]['test_mae']:.3f} mg/dL")
    print(f"✓ Test RMSE: {test_results[0]['test_rmse']:.3f} mg/dL")
    print("="*70)

if __name__ == '__main__':
    smoke_test()
