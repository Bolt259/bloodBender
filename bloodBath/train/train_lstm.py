#!/usr/bin/env python3
"""Train harness for LSTM model (minimal).

This script loads the chronological split CSVs produced earlier and runs a tiny
training loop. It supports a --max-rows parameter so we can smoke-test quickly.

Behavior:
 - Uses PyTorch if installed, otherwise uses numpy fallback model.
 - Loads `bg` and other features; replaces NaN bg with 0 for smoke runs and
   uses bg_missing_flag where appropriate.
"""
import argparse
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd

from bloodBath.models.lstm_model import build_model, TORCH_AVAILABLE

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('train')


def load_csv_subset(path: Path, max_rows: int = None):
    df = pd.read_csv(path, comment='#')
    if max_rows is not None:
        df = df.head(max_rows)
    return df


def df_to_numpy(df: pd.DataFrame):
    # Basic feature selection: bg, delta_bg, basal_rate, bolus_dose, sin_time, cos_time
    features = ['bg', 'delta_bg', 'basal_rate', 'bolus_dose', 'sin_time', 'cos_time']
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    X = df[features].fillna(0.0).to_numpy(dtype=np.float32)
    # For a simple regression target, use next-step bg (shifted)
    y = df['bg'].fillna(0.0).to_numpy(dtype=np.float32)
    return X.reshape(1, X.shape[0], X.shape[1]), y.reshape(1, y.shape[0])


def train_one_epoch_numpy(model, X, y):
    # model is NumpyLSTMStub
    loss = model.train_step(X, y)
    return loss


def train_one_epoch_torch(model, X, y):
    import torch
    import torch.nn as nn
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.from_numpy(X).float()
    # Use last-step bg as scalar target for demo
    y_t = torch.from_numpy(y[:, -1]).float().unsqueeze(1)

    optimizer.zero_grad()
    preds = model(X_t)
    loss = criterion(preds, y_t)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pump-serial', default='881235')
    parser.add_argument('--base-dir', type=Path, default=Path(__file__).parent.parent / 'bloodBank')
    parser.add_argument('--subset', choices=['train','validate','test'], default='train')
    parser.add_argument('--max-rows', type=int, default=500)
    args = parser.parse_args()

    logger = setup_logger()

    dataset_dir = args.base_dir / 'lstm_pump_data' / f'pump_{args.pump_serial}' / args.subset
    if not dataset_dir.exists():
        logger.error('Dataset not found: %s', dataset_dir)
        return

    # Find the first CSV in subset
    files = list(dataset_dir.glob('*.csv'))
    if not files:
        logger.error('No CSV files in subset: %s', dataset_dir)
        return

    csv_file = files[0]
    logger.info('Loading %s', csv_file)
    df = load_csv_subset(csv_file, max_rows=args.max_rows)

    X, y = df_to_numpy(df)
    input_size = X.shape[2]

    model = build_model(input_size)

    logger.info('Torch available: %s', TORCH_AVAILABLE)
    if TORCH_AVAILABLE:
        loss = train_one_epoch_torch(model, X, y)
    else:
        loss = train_one_epoch_numpy(model, X, y)

    logger.info('Smoke training loss: %s', loss)


if __name__ == '__main__':
    main()
