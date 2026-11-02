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
            'site_change': 0,
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
                hasattr(event, 'currentGlucoseDisplayValue') or
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
        
        # Debugging Bolus attributes
        # for i, event in enumerate(bolus_events[:5]):
        #     ts = extract_timestamp_from_event(event)
        #     dose = self._extract_bolus_dose(event)
        #     logger.info("BOLUS DEBUG %d type=%s ts=%s dose=%s", i+1, type(event).__name__, ts, dose)
        #     logger.info(
        #         "BOLUS DEBUG attrs=%s",
        #         [a for a in dir(event) if any(k in a.lower() for k in ["bolus","deliver","total","food","correct","request","insulin"])]
        #     )

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
    
    def _is_site_change_event(self, event: Any) -> bool:
        """Check if event is a cannula-fill (infusion-site change) event.

        LID_CANNULA_FILLED (event class 61) fires when the patient primes the
        new cannula after inserting a new infusion set. The parsed dataclass is
        ``LidCannulaFilled`` and carries ``NAME = "LID_CANNULA_FILLED"``; match
        on either so both parsed objects and raw-ish variants are caught.
        """
        event_type = type(event).__name__
        if event_type == 'LidCannulaFilled':
            return True
        name = getattr(event, 'NAME', None)
        return name == 'LID_CANNULA_FILLED'

    def normalize_site_change_events(self, raw_events: List[Any]) -> List[Dict[str, Any]]:
        """
        Filter raw events for LID_CANNULA_FILLED and normalize to standard format.

        When the patient changes their infusion set, insulin absorption is
        unreliable for ~2 hours ("new site lag"). LID_CANNULA_FILLED is the
        definitive signal that a new site was primed.

        Args:
            raw_events: List of raw pump events (any types)

        Returns:
            List of normalized site-change data:
            [{'timestamp': ..., 'site_change': 1, 'event_type': 'site_change'}]
        """
        normalized = []

        for event in raw_events:
            if not self._is_site_change_event(event):
                continue
            try:
                timestamp = extract_timestamp_from_event(event)
                if timestamp is not None:
                    normalized.append({
                        'timestamp': timestamp,
                        'site_change': 1,
                        'event_type': 'site_change'
                    })
                    self.event_counts['site_change'] += 1
            except Exception as e:
                logger.debug(f"Error processing site-change event: {e}")
                continue

        logger.info(f"Normalized {len(normalized)} site-change events")
        return normalized

    def _extract_bg_value(self, event: Any) -> Optional[float]:
        """Extract BG value from CGM event"""
        bg_value = None

        # Primary CGM reading is LidCgmDataGxb (id 256): the glucose value lives
        # in `currentGlucoseDisplayValue` (camelCase, mg/dL). The prior lowercase
        # lookup never matched the parsed attribute, so ~93% of readings were
        # silently dropped. Keep a lowercase fallback for any variant.
        if hasattr(event, 'currentGlucoseDisplayValue'):
            bg_value = event.currentGlucoseDisplayValue
        elif hasattr(event, 'currentglucosedisplayvalue'):
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

        # glucoseValueStatus 1=HIGH / 2=LOW zero out currentGlucoseDisplayValue;
        # map those to the Dexcom display rails rather than discarding them.
        status = getattr(event, 'glucoseValueStatusRaw', None)
        if status in (1, 2) and (bg_value is None or bg_value == 0):
            bg_value = 400 if status == 1 else 40

        if bg_value is not None:
            logger.debug(f"Extracted BG value: {bg_value} from event type: {type(event).__name__}")
        
        return float(bg_value) if bg_value is not None else None
    
    def _extract_basal_rate(self, event: Any) -> float:
        """Extract basal rate in U/hr.

        Two real basal event types reach here (from eventparser/events.py):
          LidBasalRateChange (id 3)  / LidNewDay (id 90):
              commandedBasalRate: float  # already in units/hour
          LidBasalDelivery (id 279):
              commandedRate: int         # milliunits/hour — divide by 1000

        NOTE: These fixes (commandedBasalRate probe + /1000 divisor) change the
        values produced relative to the pre-fix extractor.  The current v1.3
        artifacts (scaler.pkl / model.onnx) were trained on the pre-fix values
        (commandedRate / 100, commandedBasalRate = 0).  Do NOT restart the
        inference service with this fix until a retrain is complete — see
        TICKETS.md BB-06.
        """
        # LidBasalRateChange / LidNewDay: units/hour directly
        if hasattr(event, 'commandedBasalRate') and event.commandedBasalRate is not None:
            return float(event.commandedBasalRate)
        # LidBasalDelivery: milliunits/hour → units/hour
        if hasattr(event, 'commandedRate') and event.commandedRate is not None:
            return float(event.commandedRate) / 1000.0
        return 0.0
    
    def _extract_bolus_dose(self, event: Any) -> float:
        """Extract bolus dose in units.

        Two real bolus event types carry a dose value (from eventparser/events.py):
          LidBolusDelivery (id 279):
              deliveredTotal: int   # milliunits — divide by 1000
          LidBolusCompleted (id 20) / LidBolexCompleted (id 21):
              insulinDelivered: float  # already in units

        Both event types may fire for the same physical bolus (delivery events
        during + completion event at end).  They carry different timestamps so
        deduplicate_events() will not catch the overlap.  The resampler takes
        the max bolus value per 5-min bin, so double-counting is low-risk in
        practice, but see TICKETS.md BB-02 note on bolus double-counting audit.

        NOTE: Adding insulinDelivered changes values vs the pre-fix extractor
        which silently dropped LidBolusCompleted events — see TICKETS.md BB-06.
        """
        # LidBolusDelivery: milliunits → units
        if hasattr(event, "deliveredTotal") and event.deliveredTotal is not None:
            v = float(event.deliveredTotal)
            return v / 1000.0

        # LidBolusCompleted / LidBolexCompleted: already in units
        if hasattr(event, "insulinDelivered") and event.insulinDelivered is not None:
            return float(event.insulinDelivered)

        return 0.0
    
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
            'site_change': 0,
            'unknown': 0
        }
