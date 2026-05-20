import pandas as pd
from bloodBath.core.config import DATA_PATHS
# from  ..utils.logging_utils import 

def save_chronological_splits(
                                   lstm_df: pd.DataFrame,
                                   pump_serial: str,
                                   dataset: str,
                                   start_date: str,
                                   end_date: str) -> None:
        """Save chronological train/validate/test splits for merged LSTM datasets."""
        if lstm_df.empty:
            return

        split_df = lstm_df.copy()
        if 'timestamp' in split_df.columns:
            split_df['timestamp'] = pd.to_datetime(split_df['timestamp'], errors='coerce')
            split_df = split_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        if split_df.empty:
            return

        total = len(split_df)
        train_end = max(int(total * 0.70), 1)
        validate_end = max(train_end + int(total * 0.15), train_end)

        train_df = split_df.iloc[:train_end]
        validate_df = split_df.iloc[train_end:validate_end]
        test_df = split_df.iloc[validate_end:]

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        

        split_outputs = {
            'train': train_df,
            'validate': validate_df,
            'test': test_df,
        }

        for split_name, split_data in split_outputs.items():
            if split_data.empty:
                continue

            base_name = f"pump_{pump_serial}_{start_date}_to_{end_date}_{timestamp}.csv"
            lstm_name = f"lstm_{split_name}_{pump_serial}_{timestamp}.csv"
            split_path = DATA_PATHS['merged'][split_name]
            split_path.mkdir(parents=True, exist_ok=True)
            output_file = _chron_split_path(split_path, lstm_name, pump_serial, dataset)
            split_data.to_csv(output_file, index=False)
            print(f"Saved {split_name} split ({len(split_data)} rows) to {output_file}")
    
def _chron_split_path(merged_path, base_name, serial_number, dataset):
    split_path = merged_path.parent / dataset / f"pump_{serial_number}" / merged_path.name / base_name
    split_path.parent.mkdir(parents=True, exist_ok=True)
    return split_path