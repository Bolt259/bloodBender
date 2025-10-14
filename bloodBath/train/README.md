LSTM Training helper

Usage (smoke test):

```bash
# from repo root
bloodBath-env/bin/python bloodBath/train/train_lstm.py --pump-serial 881235 --subset train --max-rows 500
```

This will run a tiny training step using PyTorch if installed, otherwise a numpy
fallback implementation is used so the script can be executed quickly.
