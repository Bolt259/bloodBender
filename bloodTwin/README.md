# bloodTwin - LSTM Blood Glucose Prediction

**bloodTwin** is a PyTorch Lightning-based LSTM model for predicting blood glucose dynamics from continuous glucose monitoring (CGM) and insulin pump data.

## Overview

- **Model**: Multi-layer LSTM with dropout regularization
- **Input**: 24-hour lookback window (288 5-min intervals)
- **Output**: 60-minute forecast horizon (12 5-min intervals)
- **Training**: Mixed precision (AMP) with CUDA acceleration
- **Data**: Unified model trained on multiple pump datasets

## Directory Structure

```
bloodTwin/
├── models/          # LSTM model architecture
│   └── lstm.py
├── data/            # Dataset and dataloader
│   └── dataset.py
├── pipelines/       # Training and evaluation scripts
│   └── train_lstm.py
├── configs/         # YAML configurations
│   └── lstm.yaml
├── artifacts/       # Saved models, checkpoints, exports
│   └── bloodtwin_unified_lstm/
│       ├── checkpoints/
│       ├── scaler.pkl
│       ├── model.ts          # TorchScript export
│       ├── model.onnx        # ONNX export
│       └── test_results.yaml
└── analytics/       # Metrics and logs
    ├── lstm_metrics/
    └── tensorboard_logs/
```

## Features

### Input Features (8 dimensions)
1. `bg` - Blood glucose (mg/dL)
2. `delta_bg` - BG rate of change
3. `basal_rate` - Basal insulin rate (U/hr)
4. `bolus_dose` - Bolus insulin (U)
5. `sin_time` - Temporal encoding (sine)
6. `cos_time` - Temporal encoding (cosine)
7. `bg_clip_flag` - BG range flag
8. `bg_missing_flag` - Missing data flag

### Model Architecture
- **Encoder**: 2-layer LSTM (128 hidden units)
- **Decoder**: 2-layer feedforward (128 → 12)
- **Regularization**: Dropout (0.2)
- **Loss**: MAE (L1 loss for outlier robustness)
- **Metrics**: MAE, RMSE, horizon-specific MAE (30min, 60min)

### Training Configuration
- **Batch size**: 128
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau
- **Precision**: 16-bit mixed (AMP)
- **Gradient clipping**: 1.0 norm
- **Early stopping**: 5 epochs patience on val_mae

## Data Preparation

Training data is pre-split chronologically:
- **Train**: 70% (oldest data)
- **Validation**: 15% (middle)
- **Test**: 15% (newest data)

Location: `bloodBath/bloodBank/lstm_pump_data/`

### Data Format
```
pump_{serial}/
  ├── train/lstm_train_*.csv
  ├── validate/lstm_validate_*.csv
  └── test/lstm_test_*.csv
```

Each CSV contains:
- Comment header with metadata
- Columns: timestamp, bg, delta_bg, basal_rate, bolus_dose, sin_time, cos_time, bg_clip_flag, bg_missing_flag
- 5-minute temporal resolution
- Continuous time series (no gaps >15 hours)

## Usage

### 1. Environment Setup

```bash
# Activate virtual environment
source bloodBath-env/bin/activate

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Training

```bash
# Train with default config
python bloodTwin/pipelines/train_lstm.py

# Train with custom config
python bloodTwin/pipelines/train_lstm.py --config path/to/config.yaml
```

### 3. Monitor Training

```bash
# TensorBoard
tensorboard --logdir bloodTwin/analytics/tensorboard_logs
```

### 4. Artifacts

After training, the following artifacts are saved:

- `checkpoints/` - PyTorch Lightning checkpoints (.ckpt)
- `scaler.pkl` - Fitted RobustScaler for feature normalization
- `model.ts` - TorchScript export for production
- `model.onnx` - ONNX export for cross-platform deployment
- `test_results.yaml` - Final test set metrics

## Configuration

Edit `configs/lstm.yaml` to customize:

```yaml
model:
  hidden_size: 128      # LSTM hidden dimension
  num_layers: 2         # Number of LSTM layers
  dropout: 0.2          # Dropout probability

data:
  lookback: 288         # 24 hours @ 5-min
  horizon: 12           # 60 minutes forecast
  stride: 1             # Window stride
  pump_ids:             # Pumps to include
    - "881235"
    - "901161470"

training:
  batch_size: 128
  max_epochs: 50
  learning_rate: 1.0e-3
  precision: "16-mixed"  # AMP for speed
```

## Model Performance

Target metrics:
- **MAE**: <15 mg/dL (30-min horizon)
- **MAE**: <20 mg/dL (60-min horizon)
- **RMSE**: <25 mg/dL (overall)

## Export Formats

### TorchScript (.ts)
For Python inference with PyTorch:
```python
model = torch.jit.load('artifacts/bloodtwin_unified_lstm/model.ts')
predictions = model(input_tensor)
```

### ONNX (.onnx)
For cross-platform deployment:
```python
import onnxruntime as ort
session = ort.InferenceSession('artifacts/bloodtwin_unified_lstm/model.onnx')
predictions = session.run(None, {'input': input_array})
```

## Dataset Statistics

### Pump 881235
- **Records**: 393,524
- **Time range**: 2021-02-17 to 2024-10-06
- **Train**: 275,466 records (70%)
- **Val**: 59,029 records (15%)
- **Test**: 59,029 records (15%)

### Pump 901161470
- **Records**: 187,720
- **Time range**: 2024-01-13 to 2025-10-14
- **Train**: 131,404 records (70%)
- **Val**: 28,158 records (15%)
- **Test**: 28,158 records (15%)

### Combined
- **Total**: 581,244 records
- **Duration**: ~3.7 years
- **Continuous**: No gaps >15 hours

## Hardware Requirements

- **GPU**: NVIDIA RTX 2070 SUPER (8GB VRAM)
- **CUDA**: 11.8
- **PyTorch**: 2.7.1+cu118
- **Training time**: ~2-4 hours (50 epochs)
- **Inference**: ~1ms per sample (GPU)

## Physiological Constraints

- **BG range**: [20, 600] mg/dL (hard clipped)
- **Basal rate**: [0, 10] U/hr
- **Bolus**: [0, 25] U
- **Prediction clipping**: Enabled during inference

## References

- Design spec: `bloodBath/spec/bloodBath_Design_Specification_v2.0.md`
- Data preparation: `bloodBath/test_scripts/split_lstm_data_v2.py`
- Original code: `tconnectsync-bb/` (Tandem API sync)

## Future Work

- [ ] Multi-horizon training (30min, 60min, 90min)
- [ ] Ensemble models for uncertainty quantification
- [ ] Attention mechanism for interpretability
- [ ] Transfer learning for new patients
- [ ] Real-time inference pipeline
- [ ] Integration with control algorithms
