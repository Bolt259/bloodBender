"""
bloodTwin Training Pipeline

Trains unified LSTM model on multi-pump blood glucose data with GPU acceleration.
"""

import os
import sys
from pathlib import Path
import yaml
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bloodTwin.models.lstm import BloodTwinLSTM
from bloodTwin.data.dataset import create_dataloaders
from bloodTwin import ARTIFACTS_DIR, ANALYTICS_DIR, CONFIGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def train(config_path: Path):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    if 'seed' in config:
        pl.seed_everything(config['seed'], workers=True)
    
    # Paths
    data_dir = PROJECT_ROOT / config['data']['train_dir']
    artifacts_dir = ARTIFACTS_DIR / config['model']['name']
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_path = artifacts_dir / 'scaler.pkl'
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data_dir=data_dir,
        pump_ids=config['data']['pump_ids'],
        features=config['data']['features'],
        target=config['data']['target'],
        lookback=config['data']['lookback'],
        horizon=config['data']['horizon'],
        stride=config['data']['stride'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        persistent_workers=config['dataloader']['persistent_workers'],
        scaler_path=scaler_path
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = BloodTwinLSTM(
        input_size=len(config['data']['features']),
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        horizon=config['data']['horizon'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_params=config['training'].get('scheduler_params'),
        bg_min=config['constraints']['bg_min'],
        bg_max=config['constraints']['bg_max']
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Callbacks
    callbacks = []
    
    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=artifacts_dir / 'checkpoints',
        filename=config['training']['checkpoint']['filename'],
        monitor=config['training']['checkpoint']['monitor'],
        mode=config['training']['checkpoint']['mode'],
        save_top_k=config['training']['checkpoint']['save_top_k'],
        save_last=config['training']['checkpoint']['save_last'],
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # EarlyStopping (optional)
    if config['training'].get('early_stopping'):
        early_stop_callback = EarlyStopping(
            monitor=config['training']['early_stopping']['monitor'],
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode'],
            min_delta=config['training']['early_stopping']['min_delta'],
            verbose=True
        )
        callbacks.append(early_stop_callback)
        logger.info("Early stopping enabled")
    
    # LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Logger
    tensorboard_logger = None
    if config['logging']['tensorboard']:
        log_dir = PROJECT_ROOT / config['logging']['log_dir']
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_dir,
            name=config['model']['name']
        )
        logger.info(f"TensorBoard logs: {log_dir}")
    
    # Trainer
    logger.info("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['compute']['accelerator'],
        devices=config['compute']['devices'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        gradient_clip_algorithm=config['training']['gradient_clip_algorithm'],
        callbacks=callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=config['compute']['deterministic'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU detected! Training on CPU (slow)")
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test on best model
    logger.info("Testing best model...")
    test_results = trainer.test(model, test_loader, ckpt_path='best')
    
    logger.info("Test results:")
    for key, value in test_results[0].items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save test results
    results_file = artifacts_dir / 'test_results.yaml'
    with open(results_file, 'w') as f:
        yaml.dump(test_results[0], f)
    logger.info(f"Saved test results to {results_file}")
    
    # Export models if requested
    if 'export_formats' in config['artifacts']:
        logger.info("Exporting model...")
        export_model(model, config, artifacts_dir, scaler)
    
    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")


def export_model(
    model: BloodTwinLSTM,
    config: dict,
    artifacts_dir: Path,
    scaler
):
    """Export model to various formats."""
    model.eval()
    model.to('cpu')  # Export on CPU for compatibility
    
    # Create example input
    lookback = config['data']['lookback']
    input_size = len(config['data']['features'])
    example_input = torch.randn(1, lookback, input_size)
    
    export_formats = config['artifacts']['export_formats']
    
    # TorchScript
    if 'torchscript' in export_formats:
        try:
            ts_path = artifacts_dir / 'model.ts'
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(str(ts_path))
            logger.info(f"Exported TorchScript model to {ts_path}")
        except Exception as e:
            logger.error(f"Failed to export TorchScript: {e}")
    
    # ONNX
    if 'onnx' in export_formats:
        try:
            onnx_path = artifacts_dir / 'model.onnx'
            onnx_config = config['artifacts']['onnx']
            
            torch.onnx.export(
                model,
                example_input,
                str(onnx_path),
                input_names=onnx_config['input_names'],
                output_names=onnx_config['output_names'],
                dynamic_axes=onnx_config.get('dynamic_axes', None),
                opset_version=onnx_config['opset_version'],
                do_constant_folding=True
            )
            logger.info(f"Exported ONNX model to {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train bloodTwin LSTM model")
    parser.add_argument(
        '--config',
        type=Path,
        default=CONFIGS_DIR / 'lstm.yaml',
        help='Path to config YAML file'
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    train(args.config)


if __name__ == '__main__':
    main()
