# bloodBath bloodBank - Unified Data Architecture

## Directory Structure

### raw/
- `cgm/` - Continuous glucose monitoring data
- `basal/` - Basal insulin rate data  
- `bolus/` - Bolus insulin data
- `lstm/` - LSTM-ready sequence data
- `metadata/` - System metadata and logs

### merged/
- `train/` - Training datasets (70%)
- `validate/` - Validation datasets (15%)
- `test/` - Test datasets (15%)

### archives/
- `legacy/` - Archived legacy data
- `logs/` - System operation logs

## Usage

All historical LSTM and pump data has been consolidated into this unified structure.
The bloodBath package automatically references this new location.

## Schema Version: 2.0
