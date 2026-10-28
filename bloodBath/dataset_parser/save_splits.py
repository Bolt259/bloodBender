import pandas as pd
from bloodBath.core.config import DATA_PATHS


def save_chronological_splits(
        lstm_df: pd.DataFrame,
        pump_serial: str,
        dataset: str,
        start_date: str,
        end_date: str) -> None:
    """Save chronological train/validate/test splits for an external dataset.

    The split is computed once over the full chronologically-sorted series
    (70/15/15) and written to exactly one deterministic file per split. Any
    stale split files for this subject are removed first so the dataset loader
    (which globs ``lstm_{split}_*.csv``) can never concatenate independently
    split runs — that would leak a later run's train rows past an earlier
    run's test boundary.
    """
    if lstm_df.empty:
        return

    split_df = lstm_df.copy()
    if 'timestamp' in split_df.columns:
        split_df['timestamp'] = pd.to_datetime(split_df['timestamp'], errors='coerce')
        split_df = (split_df.dropna(subset=['timestamp'])
                            .sort_values('timestamp')
                            .reset_index(drop=True))

    if split_df.empty:
        return

    total = len(split_df)
    train_end = max(int(total * 0.70), 1)
    validate_end = max(train_end + int(total * 0.15), train_end)

    split_outputs = {
        'train': split_df.iloc[:train_end],
        'validate': split_df.iloc[train_end:validate_end],
        'test': split_df.iloc[validate_end:],
    }

    for split_name, split_data in split_outputs.items():
        merged_path = DATA_PATHS['merged'][split_name]
        split_dir = merged_path.parent / dataset / f"pump_{pump_serial}" / merged_path.name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Drop stale split files for this subject so the loader sees one file.
        for stale in split_dir.glob(f"lstm_{split_name}_{pump_serial}_*.csv"):
            stale.unlink()

        if split_data.empty:
            continue

        output_file = split_dir / f"lstm_{split_name}_{pump_serial}.csv"
        split_data.to_csv(output_file, index=False)
        print(f"Saved {split_name} split ({len(split_data)} rows) to {output_file}")
