"""
Data extraction and normalization for Tandem pump events
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from ..utils.time_utils import extract_timestamp_from_event
from ..core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class EventExtractor:
    """
    Extracts and normalizes raw pump events into structured data
    """
    
    def __init__(self):
        self.event_counts = {
            'cgm': 0,
            'basal': 0,
            'bolus': 0,
            'unknown': 0
        }
    
    def extract_events(self, raw_events: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract and categorize events by type
        
        Args:
            raw_events: List of raw pump events
            
        Returns:
            Dictionary with categorized events
        """
        # Handle both generator and list inputs
        if hasattr(raw_events, '__iter__') and not isinstance(raw_events, (dict, str)):
            if not isinstance(raw_events, list):
                logger.debug("Converting generator to list")
                raw_events = list(raw_events)
                logger.debug(f"Converted to list with {len(raw_events)} events")
        
        # Categorize events by type
        cgm_events = []
        basal_events = []
        bolus_events = []
        
        for event in raw_events:
            event_type = type(event).__name__
            
            # Enhanced event detection
            if self._is_cgm_event(event, event_type):
                cgm_events.append(event)
                self.event_counts['cgm'] += 1
            elif self._is_basal_event(event, event_type):
                basal_events.append(event)
                self.event_counts['basal'] += 1
            elif self._is_bolus_event(event, event_type):
                bolus_events.append(event)
                self.event_counts['bolus'] += 1
            else:
                self.event_counts['unknown'] += 1
        
        logger.debug(f"Categorized events: CGM={len(cgm_events)}, Basal={len(basal_events)}, Bolus={len(bolus_events)}")
        
        return {
            'cgm_events': cgm_events,
            'basal_events': basal_events,
            'bolus_events': bolus_events
        }
    
    def _is_cgm_event(self, event: Any, event_type: str) -> bool:
        """Check if event is a CGM event"""
        return (any(keyword in event_type for keyword in ['Cgm', 'cgm', 'Gx']) or
                hasattr(event, 'sgv') or
                hasattr(event, 'bg') or
                hasattr(event, 'currentglucosedisplayvalue') or
                hasattr(event, 'glucoseValue'))
    
    def _is_basal_event(self, event: Any, event_type: str) -> bool:
        """Check if event is a basal event"""
        return (any(keyword in event_type for keyword in ['Basal', 'basal']) or
                hasattr(event, 'commandedRate') or
                hasattr(event, 'commandedbasalrate'))
    
    def _is_bolus_event(self, event: Any, event_type: str) -> bool:
        """Check if event is a bolus event"""
        return (any(keyword in event_type for keyword in ['Bolus', 'bolus']) or
                hasattr(event, 'bolusAmount') or
                hasattr(event, 'insulin'))
    
    def normalize_cgm_events(self, cgm_events: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize CGM events to standard format
        
        Args:
            cgm_events: List of CGM events
            
        Returns:
            List of normalized CGM data
        """
        normalized = []
        
        for event in cgm_events:
            try:
                timestamp = extract_timestamp_from_event(event)
                bg_value = self._extract_bg_value(event)
                
                if timestamp is not None and bg_value is not None and bg_value > 0:
                    normalized.append({
                        'timestamp': timestamp,
                        'bg': float(bg_value),
                        'event_type': 'cgm'
                    })
            except Exception as e:
                logger.debug(f"Error processing CGM event: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized)} CGM events from {len(cgm_events)} raw events")
        return normalized
    
    def normalize_basal_events(self, basal_events: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize basal events to standard format
        
        Args:
            basal_events: List of basal events
            
        Returns:
            List of normalized basal data
        """
        normalized = []
        
        for event in basal_events:
            try:
                timestamp = extract_timestamp_from_event(event)
                basal_rate = self._extract_basal_rate(event)
                
                if timestamp is not None and basal_rate >= 0:
                    normalized.append({
                        'timestamp': timestamp,
                        'basal_rate': basal_rate,
                        'event_type': 'basal'
                    })
            except Exception as e:
                logger.debug(f"Error processing basal event: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized)} basal events from {len(basal_events)} raw events")
        return normalized
    
    def normalize_bolus_events(self, bolus_events: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize bolus events to standard format
        
        Args:
            bolus_events: List of bolus events
            
        Returns:
            List of normalized bolus data
        """
        normalized = []
        
        for event in bolus_events:
            try:
                timestamp = extract_timestamp_from_event(event)
                bolus_dose = self._extract_bolus_dose(event)
                
                if timestamp is not None and bolus_dose > 0:
                    normalized.append({
                        'timestamp': timestamp,
                        'bolus_dose': bolus_dose,
                        'event_type': 'bolus'
                    })
            except Exception as e:
                logger.debug(f"Error processing bolus event: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized)} bolus events from {len(bolus_events)} raw events")
        return normalized
    
    def _extract_bg_value(self, event: Any) -> Optional[float]:
        """Extract BG value from CGM event"""
        bg_value = None
        
        # Check for currentglucosedisplayvalue (used by LidCgmDataG7)
        if hasattr(event, 'currentglucosedisplayvalue'):
            bg_value = event.currentglucosedisplayvalue
        elif hasattr(event, 'sgv'):
            bg_value = event.sgv
        elif hasattr(event, 'bg'):
            bg_value = event.bg
        elif hasattr(event, 'value'):
            bg_value = event.value
        elif hasattr(event, 'glucoseValue'):
            bg_value = event.glucoseValue
        else:
            logger.debug(f"No BG value found in CGM event type: {type(event).__name__}, attributes: {dir(event)}")
        
        if bg_value is not None:
            logger.debug(f"Extracted BG value: {bg_value} from event type: {type(event).__name__}")
        
        return float(bg_value) if bg_value is not None else None
    
    def _extract_basal_rate(self, event: Any) -> float:
        """Extract basal rate with proper unit conversion"""
        basal_rate = 0.0
        
        if hasattr(event, 'commandedRate') and event.commandedRate is not None:
            # commandedRate is in hundredths of units/hr
            basal_rate = float(event.commandedRate) / 100.0
        elif hasattr(event, 'commandedbasalrate') and event.commandedbasalrate is not None:
            basal_rate = float(event.commandedbasalrate)
        elif hasattr(event, 'value') and event.value is not None:
            value = float(event.value)
            basal_rate = value / 100.0 if value > 10 else value
        
        return basal_rate
    
    def _extract_bolus_dose(self, event: Any) -> float:
        """Extract bolus dose with proper unit conversion"""
        bolus_dose = 0.0
        
        if hasattr(event, 'bolusAmount') and event.bolusAmount is not None:
            # bolusAmount is in hundredths of units
            bolus_dose = float(event.bolusAmount) / 100.0
        elif hasattr(event, 'insulin') and event.insulin is not None:
            insulin = float(event.insulin)
            bolus_dose = insulin / 100.0 if insulin > 10 else insulin
        elif hasattr(event, 'value') and event.value is not None:
            value = float(event.value)
            bolus_dose = value / 100.0 if value > 10 else value
        
        return bolus_dose
    
    def deduplicate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate events based on timestamp and event type
        
        Args:
            events: List of events to deduplicate
            
        Returns:
            List of deduplicated events
        """
        if not events:
            return events
        
        # Convert to DataFrame for easier deduplication
        df = pd.DataFrame(events)
        
        # Remove duplicates based on timestamp and event_type, keeping the last occurrence
        df_dedup = df.drop_duplicates(subset=['timestamp', 'event_type'], keep='last')
        
        # Sort by timestamp
        df_dedup = df_dedup.sort_values('timestamp')
        
        logger.info(f"Deduplicated {len(events)} events to {len(df_dedup)} events")
        
        return df_dedup.to_dict('records')  # type: ignore
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get extraction statistics
        
        Returns:
            Dictionary with extraction statistics
        """
        return {
            'event_counts': self.event_counts.copy(),
            'total_processed': sum(self.event_counts.values())
        }
    
    def reset_stats(self):
        """Reset extraction statistics"""
        self.event_counts = {
            'cgm': 0,
            'basal': 0,
            'bolus': 0,
            'unknown': 0
        }
