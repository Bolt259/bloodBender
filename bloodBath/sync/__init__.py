"""
bloodBath.sync - Data synchronization and harvesting module

This module provides comprehensive data synchronization capabilities for bloodBath,
including production data harvesting, incremental sync, and data validation.
"""

try:
    from .harvest_manager import HarvestManager
    from .sync_engine import SyncEngine
    __all__ = ['HarvestManager', 'SyncEngine']
except ImportError as e:
    # Graceful fallback if dependencies are missing
    import logging
    logging.warning(f"bloodBath sync module import error: {e}")
    
    # Provide minimal interface
    class SyncEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("SyncEngine not available due to import errors")
    
    class HarvestManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("HarvestManager not available due to import errors")
    
    __all__ = ['HarvestManager', 'SyncEngine']