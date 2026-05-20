"""
Command-line interface for bloodBath pump synchronization
"""

import argparse
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

from ..core import (
    TandemHistoricalSyncClient, 
    PumpConfig, 
    load_pump_configs, 
    create_default_pump_configs,
    get_credentials,
    get_default_pump_serial
)
from ..core.config import DATA_PATHS, BLOODBANK_ROOT
from ..utils.logging_utils import setup_logger
from ..utils.env_utils import (
    create_env_template, 
    get_env_file_locations
)
from ..utils.pump_info import analyze_pump_activity, print_pump_summary
# DEPRECATED: sweetBlood module removed in favor of bloodBank v2.0 architecture
# from ..sweetBlood import add_sweetblood_args, handle_sweetblood_commands, SweetBloodIntegration
from ..utils.structure_utils import setup_sweetblood_environment

# Compatibility stubs for legacy sweetBlood functionality
def add_sweetblood_args(parser):
    """DEPRECATED: Legacy sweetBlood args - now handled by bloodBank v2.0"""
    pass

def handle_sweetblood_commands(args, client):
    """DEPRECATED: Legacy sweetBlood commands - now handled by bloodBank v2.0"""
    return False

class SweetBloodIntegration:
    """DEPRECATED: Legacy sweetBlood integration - now handled by bloodBank v2.0"""
    
    def prepare_training_data(self, *args, **kwargs):
        return {"status": "deprecated", "message": "Use bloodBank v2.0 architecture"}
    
    def get_training_data_info(self):
        return {"status": "deprecated", "message": "Use bloodBank v2.0 architecture"}
    
    def get_scaler_info(self):
        return {"status": "deprecated", "message": "Use bloodBank v2.0 architecture"}

logger = logging.getLogger(__name__)

DEFAULT_PUMP_SERIALS = ['881235', '901161470', '1539514']


def setup_cli_logging(verbose: bool = False, structure: Optional[Dict[str, Path]] = None):
    """Setup logging for CLI"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Use structured logs directory if available, otherwise fall back to internal logs
    if structure and 'logs' in structure:
        log_dir = structure['logs']
    else:
        log_dir = Path(__file__).parent.parent / 'logs'
    
    log_dir.mkdir(exist_ok=True)
    setup_logger(
        name='bloodBath',
        level=log_level,
        log_file=log_dir / 'bloodBath.log'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description='bloodBath: Tandem pump historical data synchronization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create .env template file for credentials
  python -m bloodBath.cli.main create-env

    # Full sync (default): all discovered pumps, full available range, chunked
    python -m bloodBath.cli.main sync

    # Sync specific pumps across full available ranges
    python -m bloodBath.cli.main sync --pump-serials 881235 901161470

    # Sync specific range for a pump
    python -m bloodBath.cli.main sync --pump-serial 123456 --start-date 2024-01-01 --end-date 2024-06-30

    # See available ranges and what sync would pull
    python -m bloodBath.cli.main available-data

  # Generate unified LSTM sequences with new pipeline
  python -m bloodBath.cli.main unified-lstm --pump-serial 123456

  # Test API connection
  python -m bloodBath.cli.main test

  # Check sync status
  python -m bloodBath.cli.main status
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--email',
        help='t:connect email address (can also use .env file or environment variables)'
    )
    
    parser.add_argument(
        '--password',
        help='t:connect password (can also use .env file or environment variables)'
    )
    
    parser.add_argument(
        '--region',
        choices=['US', 'EU'],
        default='US',
        help='Server region (default: US)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync pump data')
    sync_parser.add_argument(
        '--config',
        help='Path to pump configuration JSON file'
    )
    sync_parser.add_argument(
        '--pump-serial',
        help='Single pump serial number to sync'
    )
    sync_parser.add_argument(
        '--pump-serials',
        nargs='+',
        help='One or more pump serial numbers to sync'
    )
    sync_parser.add_argument(
        '--start-date',
        help='Start date for sync (YYYY-MM-DD format)'
    )
    sync_parser.add_argument(
        '--end-date',
        help='End date for sync (YYYY-MM-DD format, defaults to today)'
    )
    sync_parser.add_argument(
        '--update',
        action='store_true',
        help='Sync only new data since last successful sync'
    )
    sync_parser.add_argument(
        '--full-sync',
        action='store_true',
        help='Force full-range sync (default behavior)'
    )
    sync_parser.add_argument(
        '--force-full',
        action='store_true',
        help='Force full sync, ignoring existing data'
    )
    sync_parser.add_argument(
        '--parallel',
        action='store_true',
        help='Sync multiple pumps in parallel'
    )
    sync_parser.add_argument(
        '--chunk-days',
        type=int,
        default=15,
        help='Number of days per API chunk (default: 15)'
    )
    sync_parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
        help='Maximum retry attempts (default: 5)'
    )
    available_data_parser = subparsers.add_parser('available-data', help='Show available per-pump data ranges and planned sync ranges')
    available_data_parser.add_argument('--pump-serial', help='Single pump serial number filter')
    available_data_parser.add_argument('--pump-serials', nargs='+', help='Multiple pump serial number filter')
    available_data_parser.add_argument('--start-date', help='Optional planned sync start date override (YYYY-MM-DD)')
    available_data_parser.add_argument('--end-date', help='Optional planned sync end date override (YYYY-MM-DD)')
    
    # LSTM command
    lstm_parser = subparsers.add_parser('lstm', help='Generate LSTM-ready dataset')
    lstm_parser.add_argument(
        '--pump-serial',
        help='Pump serial number (can also use PUMP_SERIAL_NUMBER env var)'
    )
    lstm_parser.add_argument(
        '--output-file',
        help='Output file path (optional)'
    )
    
    # Unified LSTM command
    unified_lstm_parser = subparsers.add_parser('unified-lstm', help='Generate unified LSTM sequences with new processing pipeline')
    unified_lstm_parser.add_argument(
        '--pump-serial',
        help='Pump serial number (can also use PUMP_SERIAL_NUMBER env var)'
    )
    unified_lstm_parser.add_argument(
        '--source-dir',
        help='Source directory containing basal, bolus, and cgm CSV files'
    )
    unified_lstm_parser.add_argument(
        '--output-dir',
        default='unified_lstm_training',
        help='Output directory for LSTM sequences (default: unified_lstm_training)'
    )
    unified_lstm_parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Validate sequences for LSTM training readiness (default: True)'
    )
    unified_lstm_parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation and output all sequences'
    )
    unified_lstm_parser.add_argument(
        '--max-gap-hours',
        type=float,
        default=15.0,
        help='Maximum gap hours before sequence break (default: 15.0)'
    )
    unified_lstm_parser.add_argument(
        '--min-segment-length',
        type=int,
        default=12,
        help='Minimum sequence length in intervals (default: 12)'
    )
    unified_lstm_parser.add_argument(
        '--normalization-method',
        choices=['z-score', 'min-max'],
        default='z-score',
        help='Feature normalization method (default: z-score)'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show sync status')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create default configuration')
    config_parser.add_argument(
        '--output',
        default='pump_configs.json',
        help='Output file path (default: pump_configs.json)'
    )
    config_parser.add_argument(
        '--pump-serials',
        nargs='+',
        help='Pump serial numbers to include'
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test API connection')
    
    # Create .env template command
    env_parser = subparsers.add_parser('create-env', help='Create .env template file')
    env_parser.add_argument(
        '--output',
        default='.env',
        help='Output file for .env template (default: .env)'
    )
    env_parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing .env file'
    )
    
    # Add sweetBlood LSTM training arguments
    add_sweetblood_args(parser)
    
    return parser


def _resolve_target_serials(args, discovered_serials: Optional[List[str]] = None) -> List[str]:
    """Resolve target pump serial list from args/discovery/defaults."""
    explicit = []
    if getattr(args, 'pump_serial', None):
        explicit.append(str(args.pump_serial))
    if getattr(args, 'pump_serials', None):
        explicit.extend([str(serial) for serial in args.pump_serials])

    if explicit:
        return list(dict.fromkeys(explicit))

    if discovered_serials:
        return list(dict.fromkeys(discovered_serials))

    return list(DEFAULT_PUMP_SERIALS)


def _build_pump_configs_for_sync(args, client) -> List[PumpConfig]:
    """Build pump configs with default full-range behavior for selected pumps."""
    pumps_info = analyze_pump_activity(client, args.region)
    discovered_serials = list(pumps_info.keys()) if pumps_info else []
    target_serials = _resolve_target_serials(args, discovered_serials)

    if not target_serials:
        print("Error: No pumps resolved for sync")
        sys.exit(1)

    configs = []
    for serial in target_serials:
        user_start = args.start_date
        user_end = args.end_date

        if user_start and user_end:
            start_date = user_start
            end_date = user_end
        elif serial in pumps_info and pumps_info[serial].get('first_event') and pumps_info[serial].get('last_event'):
            available_start = pumps_info[serial]['first_event'].strftime('%Y-%m-%d')
            available_end = pumps_info[serial]['last_event'].strftime('%Y-%m-%d')
            start_date = user_start or available_start
            end_date = user_end or available_end
        else:
            end_date = user_end or datetime.now().strftime('%Y-%m-%d')
            start_date = user_start or (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            print(f"⚠️  Could not detect full range for pump {serial}; using {start_date} to {end_date}")

        configs.append(PumpConfig(serial=serial, start_date=start_date, end_date=end_date))

    return configs


def cmd_sync(args):
    """Handle sync command"""
    setup_cli_logging(args.verbose)
    
    # Get credentials from environment first, then CLI args
    creds = get_credentials(args.email, args.password, args.region)
    
    if not creds.is_valid():
        print("Error: Missing credentials")
        print("Please provide credentials via:")
        print("  1. Command line: --email <email> --password <password>")
        print("  2. Environment variables: TCONNECT_EMAIL, TCONNECT_PASSWORD")
        print("  3. .env file (use 'python -m bloodBath create-env' to create template)")
        return
    
    # Setup bloodBank data structure
    print("Using bloodBank unified data structure...")
    print(f"Data location: {BLOODBANK_ROOT}")
    
    # Ensure directories exist
    for category, paths in DATA_PATHS.items():
        if isinstance(paths, dict):
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
    
    # Initialize client with standard bloodBank layout
    client = TandemHistoricalSyncClient(
        email=creds.email,
        password=creds.password,
        region=creds.region,
        chunk_days=args.chunk_days,
        max_retries=args.max_retries
    )
    
    # Test connection first
    if not client.test_connection():
        print("Error: Unable to connect to Tandem API")
        sys.exit(1)
    
    if args.config:
        configs = load_pump_configs(args.config)
    else:
        configs = _build_pump_configs_for_sync(args, client)

    if not configs:
        print("No pumps found for synchronization")
        sys.exit(0)

    update_mode = args.update and not args.full_sync
    
    print(f"Starting sync for {len(configs)} pump(s)...")
    print(f"Data will be organized in bloodBank structure: {BLOODBANK_ROOT}")
    
    # Show pump activity summary if verbose
    if args.verbose:
        try:
            print("\n🔍 Analyzing pump activity...")
            pumps_info = analyze_pump_activity(client, creds.region)
            if pumps_info:
                print_pump_summary(pumps_info)
        except Exception as e:
            logger.warning(f"Could not analyze pump activity: {e}")
    
    # Sync pumps
    results = client.sync_multiple_pumps(
        configs,
        update_mode=update_mode,
        force_full=args.force_full,
        parallel=args.parallel
    )
    
    # Report results
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    print(f"\nSync completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed pumps:")
        for serial, success in results.items():
            if not success:
                print(f"  - {serial}")
    
    # Show final stats
    stats = client.get_client_stats()
    print(f"\nStatistics:")
    print(f"  Total events processed: {stats['extraction_stats']['total_processed']}")
    print(f"  CGM events: {stats['extraction_stats']['event_counts']['cgm']}")
    print(f"  Basal events: {stats['extraction_stats']['event_counts']['basal']}")
    print(f"  Bolus events: {stats['extraction_stats']['event_counts']['bolus']}")
    
    print(f"\nData organized in bloodBank structure:")
    print(f"  Raw CGM directory: {DATA_PATHS['raw']['cgm']}")
    print(f"  Raw Basal directory: {DATA_PATHS['raw']['basal']}")
    print(f"  Raw Bolus directory: {DATA_PATHS['raw']['bolus']}")
    print(f"  Raw LSTM directory: {DATA_PATHS['raw']['lstm']}")
    print(f"  Training directory: {DATA_PATHS['merged']['train']}")
    print(f"  Validation directory: {DATA_PATHS['merged']['validate']}")
    print(f"  Test directory: {DATA_PATHS['merged']['test']}")
    
    client.disconnect()


def cmd_lstm(args):
    """Handle LSTM command"""
    setup_cli_logging(args.verbose)
    
    # Get pump serial from args or environment
    pump_serial = args.pump_serial or get_default_pump_serial()
    
    if not pump_serial:
        print("Error: Must specify --pump-serial or set PUMP_SERIAL_NUMBER env var")
        sys.exit(1)
    
    # Use sweetBlood integration for LSTM operations
    sweetblood_integration = SweetBloodIntegration()
    
    # Prepare LSTM training data
    result = sweetblood_integration.prepare_training_data(
        pump_serial=pump_serial,
        sequence_length=getattr(args, 'sequence_length', 60),
        prediction_horizon=getattr(args, 'prediction_horizon', 1),
        test_split=getattr(args, 'test_split', 0.2),
        val_split=getattr(args, 'val_split', 0.2),
        save_data=True
    )
    try:
        print("LSTM training data prepared successfully!")
        print(f"Training samples: {result['metadata']['train_samples']}")
        print(f"Validation samples: {result['metadata']['val_samples']}")
        print(f"Test samples: {result['metadata']['test_samples']}")
    
    
        if 'saved_files' in result and result['saved_files']:
            saved_files = result['saved_files']
            if isinstance(saved_files, dict) and 'train_X' in saved_files:
                train_file = saved_files['train_X']
                if isinstance(train_file, Path):
                    print(f"Data saved to: {train_file.parent}")
                else:
                    print("Training data prepared (in-memory)")
            else:
                print("Training data prepared (in-memory)")
        else:
            print("Training data prepared (in-memory)")
    
    except KeyError as e:
        print(f"Warning: Missing expected metadata key: {e}")
        print("Result may be incomplete or in unexpected format.")
        print(f"Full result: {result}")


def cmd_unified_lstm(args):
    """Handle unified LSTM processing command"""
    setup_cli_logging(args.verbose)
    
    from ..data.processors import UnifiedDataProcessor
    from ..data.validators import LstmDataValidator
    
    # Get pump serial from args or environment
    pump_serial = args.pump_serial or get_default_pump_serial()
    
    if not pump_serial:
        print("Error: Must specify --pump-serial or set PUMP_SERIAL_NUMBER env var")
        sys.exit(1)
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    pump_output_dir = output_dir / f"pump_{pump_serial}"
    logs_dir = output_dir / "logs"
    validation_dir = output_dir / "validation_reports"
    
    # Create directories
    pump_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("UNIFIED LSTM PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Pump Serial: {pump_serial}")
    print(f"Source Directory: {args.source_dir}")
    print(f"Output Directory: {pump_output_dir}")
    print(f"Max Gap Hours: {args.max_gap_hours}")
    print(f"Min Segment Length: {args.min_segment_length}")
    print(f"Normalization: {args.normalization_method}")
    # Determine validation setting
    do_validation = args.validate and not args.no_validate
    print(f"Validation: {do_validation}")
    
    # Initialize unified LSTM processor with specified parameters
    processor = UnifiedDataProcessor(
        freq='5min',
        max_gap_hours=args.max_gap_hours,
        max_impute_minutes=60,
        min_segment_length=args.min_segment_length,
        normalization_method=args.normalization_method
    )
    
    # Initialize validator if requested
    validator = LstmDataValidator() if do_validation else None
    
    try:
        # Determine source directory
        if args.source_dir:
            source_dir = Path(args.source_dir)
        else:
            # Use testing data directory as default
            source_dir = BLOODBANK_ROOT / 'raw' / 'lstm' / f"lstm_pump_data"
            print(f"No source directory specified, using default: {source_dir}")
        
        if not source_dir.exists():
            print(f"Error: Source directory not found: {source_dir}")
            sys.exit(1)
        
        print(f"\nProcessing pump {pump_serial} from {source_dir}...")
        
        # Use the processor's batch processing method
        result = processor.process_pump_data_for_lstm(
            pump_data_dir=source_dir,
            output_dir=pump_output_dir,
            pump_id=pump_serial,
            target_months=None,  # Process all available months
            validate_output=do_validation
        )
        
        # Report results
        try:
            print(f"\nProcessing Results:")
            print(f"  Status: {result['status']}")
            print(f"  Months Processed: {result['months_successful']}/{result['months_total']}")
            print(f"  Total Sequences: {result['total_sequences']}")
            print(f"  Valid Sequences: {result['valid_sequences']}")
            print(f"  Success Rate: {result['success_rate']:.1%}")
            print(f"  Files Saved: {len(result['saved_files'])}")

            if result['processing_errors']:
                print(f"\nProcessing Errors:")
                for error in result['processing_errors']:
                    print(f"  - {error}")
        
            # Show validation reports if available
            if 'validation_reports' in result and result['validation_reports']:
                print(f"\nValidation Reports:")
                for month, validation_report in result['validation_reports'].items():
                    summary = validation_report.get('validation_summary', {})
                    print(f"  Month {month}:")
                    print(f"    Sequences validated: {summary.get('total_sequences', 0)}")
                    print(f"    Sequences passed: {summary.get('passed_sequences', 0)}")
                    print(f"    Pass rate: {summary.get('pass_rate', 0):.1%}")

        except KeyError as e:
            print(f"Warning: Missing expected result key: {e}")
            print("Result may be incomplete or in unexpected format.")
            print(f"Full result: {result}")
        
        # Verify output files
        print(f"\nVerifying output files...")
        csv_files = list(pump_output_dir.glob("**/*.csv"))
        json_files = list(pump_output_dir.glob("**/*metadata.json"))
        
        print(f"  CSV files: {len(csv_files)}")
        print(f"  JSON metadata files: {len(json_files)}")
        
        # Sample file validation
        if csv_files:
            sample_csv = csv_files[0]
            print(f"  Sample file: {sample_csv.name}")
            
            # Quick format check
            try:
                import pandas as pd
                df = pd.read_csv(sample_csv, comment='#')
                expected_columns = ['timestamp', 'bg', 'basal', 'bolus', 'features', 'mask_bg', 'mask_label']
                
                if df.columns.tolist() == expected_columns:
                    print(f"  ✓ Correct CSV format with {len(df)} rows")
                else:
                    print(f"  ⚠ Unexpected columns: {df.columns.tolist()}")
                
            except Exception as e:
                print(f"  ⚠ Error reading sample file: {e}")
        
        print(f"\n" + "=" * 80)
        print("UNIFIED LSTM PROCESSING COMPLETED")
        print("=" * 80)
        print(f"Output directory: {pump_output_dir}")
        print(f"Total valid sequences: {result['valid_sequences']}")
        print(f"Ready for LSTM training: {'Yes' if result['valid_sequences'] > 0 else 'No'}")
        
        if result['valid_sequences'] > 0:
            print(f"\nNext steps:")
            print(f"1. Load CSV files for LSTM training")
            print(f"2. Parse 'features' JSON column for additional inputs")
            print(f"3. Use 'mask_bg' and 'mask_label' for proper masking")
            print(f"4. Create sequences with sliding window approach")
        
        return result['status'] == 'completed'
        
    except Exception as e:
        logger.error(f"Unified LSTM processing failed: {e}")
        print(f"Error: {e}")
        return False


def cmd_status(args):
    """Handle status command"""
    setup_cli_logging(args.verbose)
    
    # Use sweetBlood integration for status
    sweetblood_integration = SweetBloodIntegration()
    
    # Get training data info
    training_info = sweetblood_integration.get_training_data_info()
    try:
        print("sweetBlood Training Data Status:")
        print(f"  Available datasets: {training_info['total_count']}")
        if training_info['latest_timestamp']:
            print(f"  Latest dataset: {training_info['latest_timestamp']}")
        print(f"  Data directory: {training_info['training_data_dir']}")
        
        # Get scaler info
        scaler_info = sweetblood_integration.get_scaler_info()
        if scaler_info:
            print("\nScaler Information:")
            for name, details in scaler_info.items():
                print(f"  {name}: {details['type']}")
        
        
    except Exception as e:
        logger.error(f"Error retrieving status: {e}")
        print(f"Error retrieving status: {e}")
        print(f"Full Training Info: {training_info}")

    # Initialize client for pump status
    internal_sweetblood_path = str(Path(__file__).parent.parent / 'bloodBank')
    client = TandemHistoricalSyncClient(
        output_dir=str(setup_sweetblood_environment(internal_sweetblood_path)['lstm_pump_data'])
    )

    status = client.get_sync_status()
    
    if not status:
        print("No sync data found")
        return
    
    print("Sync Status:")
    print("=" * 60)
    
    for serial, pump_status in status.items():
        print(f"\nPump {serial}:")
        print(f"  Last sync: {pump_status['last_successful_sync'] or 'Never'}")
        print(f"  Total records: {pump_status['total_records']}")
        print(f"  Last updated: {pump_status['last_updated'] or 'Never'}")
        print(f"  Failed ranges: {pump_status['failed_ranges']}")
        print(f"  CSV files: {pump_status['csv_files']}")


def cmd_create_config(args):
    """Handle create-config command"""
    setup_cli_logging(args.verbose)
    
    # Create default configurations
    if args.pump_serials:
        configs = []
        for serial in args.pump_serials:
            config = PumpConfig(
                serial=serial,
                start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            configs.append(config)
    else:
        configs = create_default_pump_configs()
    
    # Save to file
    output_path = Path(args.output)
    
    config_data = {
        'pumps': [
            {
                'serial': config.serial,
                'start_date': config.start_date,
                'end_date': config.end_date
            }
            for config in configs
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Created configuration file: {output_path}")
    print(f"Contains {len(configs)} pump configuration(s)")


def cmd_create_env(args):
    """Create .env template file"""
    setup_cli_logging(args.verbose)
    
    output_path = Path(args.output)
    
    # Check if file exists and --force not provided
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        sys.exit(1)
    
    # Create the .env template
    try:
        created_path = create_env_template(output_path)
        print(f"✓ Created .env template at: {created_path}")
        
        # Show search locations
        print(f"\nThe .env file will be searched in the following locations:")
        for i, location in enumerate(get_env_file_locations(), 1):
            print(f"  {i}. {location}")
        
        print(f"\nEdit {created_path} with your credentials:")
        print("  - TCONNECT_EMAIL=your@email.com")
        print("  - TCONNECT_PASSWORD=your_password")
        print("  - TCONNECT_REGION=US (or EU)")
        print("  - TIMEZONE_NAME=America/Los_Angeles (or your timezone)")
        print("  - PUMP_SERIAL_NUMBER=881235 (your pump serial)")
        print("  - CACHE_CREDENTIALS=true (optional)")
        
        # Check if credentials can be validated
        print(f"\nAfter editing, you can test credentials with:")
        print("  python -m bloodBath test")
        
        print(f"\nDefault pump serial can be used with:")
        print("  python -m bloodBath sync  # Uses PUMP_SERIAL_NUMBER from .env")
        print("  python -m bloodBath lstm  # Uses PUMP_SERIAL_NUMBER from .env")
        
    except Exception as e:
        print(f"Error creating .env template: {e}")
        sys.exit(1)


def cmd_test(args):
    """Test API connection"""
    setup_cli_logging(args.verbose)
    
    # Get credentials from environment first, then CLI args
    creds = get_credentials(args.email, args.password, args.region)
    
    if not creds.is_valid():
        print("Error: Missing credentials")
        print("Please provide credentials via:")
        print("  1. Command line: --email <email> --password <password>")
        print("  2. Environment variables: TCONNECT_EMAIL, TCONNECT_PASSWORD")
        print("  3. .env file (use 'python -m bloodBath create-env' to create template)")
        sys.exit(1)
    
    client = TandemHistoricalSyncClient(
        email=creds.email,
        password=creds.password,
        region=creds.region,
    )
    
    print("Testing API connection...")
    
    if client.test_connection():
        print("✓ Connection successful")
        
        # Get connection info
        info = client.get_client_stats()['connector_info']
        print(f"  Region: {info.get('region', 'Unknown')}")
        print(f"  Connected: {info.get('connected', False)}")
    else:
        print("✗ Connection failed")
        sys.exit(1)


def cmd_available_data(args):
    """Show available per-pump data ranges and planned sync range."""
    setup_cli_logging(args.verbose)

    creds = get_credentials(args.email, args.password, args.region)
    if not creds.is_valid():
        print("Error: Missing credentials")
        print("Please provide credentials via CLI args, env vars, or .env file")
        sys.exit(1)

    client = TandemHistoricalSyncClient(
        email=creds.email,
        password=creds.password,
        region=creds.region,
    )

    if not client.test_connection():
        print("Error: Unable to connect to Tandem API")
        sys.exit(1)

    pumps_info = analyze_pump_activity(client, creds.region)
    serials = _resolve_target_serials(args, list(pumps_info.keys()))

    print("=" * 80)
    print("AVAILABLE PUMP DATA RANGES")
    print("=" * 80)

    for serial in serials:
        info = pumps_info.get(serial)
        if not info or not info.get('first_event') or not info.get('last_event'):
            print(f"\nPump {serial}")
            print("  Available range: unknown")
            print("  Planned sync range: fallback 90-day window")
            continue

        available_start = info['first_event'].strftime('%Y-%m-%d')
        available_end = info['last_event'].strftime('%Y-%m-%d')
        planned_start = args.start_date or available_start
        planned_end = args.end_date or available_end

        print(f"\nPump {serial}")
        print(f"  Available range: {available_start} to {available_end}")
        print(f"  Planned sync range: {planned_start} to {planned_end}")
        print(f"  Status: {info.get('status', 'Unknown')}")

    client.disconnect()


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle sweetBlood commands first
    if handle_sweetblood_commands(args, None):
        return
    
    # Route to appropriate command handler
    if args.command == 'sync':
        cmd_sync(args)
    elif args.command == 'lstm':
        cmd_lstm(args)
    elif args.command == 'unified-lstm':
        cmd_unified_lstm(args)
    elif args.command == 'available-data':
        cmd_available_data(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'create-config':
        cmd_create_config(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'create-env':
        cmd_create_env(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()