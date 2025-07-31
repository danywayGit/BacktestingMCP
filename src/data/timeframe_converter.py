"""
Timeframe converter for compressing OHLCV data between different timeframes.
"""

import pandas as pd
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from config.settings import TimeFrame, TIMEFRAME_MAPPING


class TimeframeConverter:
    """Convert OHLCV data between different timeframes."""
    
    # Timeframe hierarchy (smaller to larger)
    TIMEFRAME_HIERARCHY = [
        TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30,
        TimeFrame.H1, TimeFrame.H4, TimeFrame.H12, TimeFrame.D1, TimeFrame.W1
    ]
    
    @staticmethod
    def can_convert(source_tf: TimeFrame, target_tf: TimeFrame) -> bool:
        """Check if conversion is possible (source must be smaller than target)."""
        try:
            source_idx = TimeframeConverter.TIMEFRAME_HIERARCHY.index(source_tf)
            target_idx = TimeframeConverter.TIMEFRAME_HIERARCHY.index(target_tf)
            return source_idx < target_idx
        except ValueError:
            return False
    
    @staticmethod
    def convert_timeframe(
        data: pd.DataFrame,
        source_tf: TimeFrame,
        target_tf: TimeFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Convert OHLCV data from one timeframe to another.
        
        Args:
            data: DataFrame with OHLCV data (timestamp index)
            source_tf: Source timeframe
            target_tf: Target timeframe
            validate: Validate conversion is possible
            
        Returns:
            DataFrame with converted OHLCV data
        """
        if validate and not TimeframeConverter.can_convert(source_tf, target_tf):
            raise ValueError(f"Cannot convert from {source_tf} to {target_tf}")
        
        if data.empty:
            return data.copy()
        
        # Get pandas frequency string for target timeframe
        target_freq = TIMEFRAME_MAPPING[target_tf]
        
        # Ensure the data is properly indexed with datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")
        
        # Required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Group by target timeframe and aggregate
        grouper = pd.Grouper(freq=target_freq, label='left', closed='left')
        
        agg_dict = {
            'open': 'first',    # First open of the period
            'high': 'max',      # Highest high of the period
            'low': 'min',       # Lowest low of the period
            'close': 'last',    # Last close of the period
            'volume': 'sum'     # Sum of volumes
        }
        
        # Add any additional columns (preserve with last value)
        for col in data.columns:
            if col not in required_cols:
                agg_dict[col] = 'last'
        
        converted = data.groupby(grouper).agg(agg_dict)
        
        # Remove rows with NaN values (incomplete periods)
        converted = converted.dropna()
        
        return converted
    
    @staticmethod
    def convert_multiple_timeframes(
        data: pd.DataFrame,
        source_tf: TimeFrame,
        target_timeframes: List[TimeFrame]
    ) -> Dict[TimeFrame, pd.DataFrame]:
        """Convert data to multiple target timeframes."""
        results = {}
        
        for target_tf in target_timeframes:
            if TimeframeConverter.can_convert(source_tf, target_tf):
                results[target_tf] = TimeframeConverter.convert_timeframe(
                    data, source_tf, target_tf
                )
        
        return results
    
    @staticmethod
    def get_convertible_timeframes(source_tf: TimeFrame) -> List[TimeFrame]:
        """Get list of timeframes that the source can be converted to."""
        try:
            source_idx = TimeframeConverter.TIMEFRAME_HIERARCHY.index(source_tf)
            return TimeframeConverter.TIMEFRAME_HIERARCHY[source_idx + 1:]
        except ValueError:
            return []
    
    @staticmethod
    def estimate_bars_count(
        source_tf: TimeFrame,
        target_tf: TimeFrame,
        source_bars: int
    ) -> int:
        """Estimate number of bars after conversion."""
        # Conversion ratios (approximate)
        conversion_ratios = {
            (TimeFrame.M1, TimeFrame.M5): 5,
            (TimeFrame.M1, TimeFrame.M15): 15,
            (TimeFrame.M1, TimeFrame.M30): 30,
            (TimeFrame.M1, TimeFrame.H1): 60,
            (TimeFrame.M1, TimeFrame.H4): 240,
            (TimeFrame.M1, TimeFrame.H12): 720,
            (TimeFrame.M1, TimeFrame.D1): 1440,
            (TimeFrame.M5, TimeFrame.M15): 3,
            (TimeFrame.M5, TimeFrame.M30): 6,
            (TimeFrame.M5, TimeFrame.H1): 12,
            (TimeFrame.M5, TimeFrame.H4): 48,
            (TimeFrame.M5, TimeFrame.H12): 144,
            (TimeFrame.M5, TimeFrame.D1): 288,
            (TimeFrame.M15, TimeFrame.M30): 2,
            (TimeFrame.M15, TimeFrame.H1): 4,
            (TimeFrame.M15, TimeFrame.H4): 16,
            (TimeFrame.M15, TimeFrame.H12): 48,
            (TimeFrame.M15, TimeFrame.D1): 96,
            (TimeFrame.M30, TimeFrame.H1): 2,
            (TimeFrame.M30, TimeFrame.H4): 8,
            (TimeFrame.M30, TimeFrame.H12): 24,
            (TimeFrame.M30, TimeFrame.D1): 48,
            (TimeFrame.H1, TimeFrame.H4): 4,
            (TimeFrame.H1, TimeFrame.H12): 12,
            (TimeFrame.H1, TimeFrame.D1): 24,
            (TimeFrame.H1, TimeFrame.W1): 168,
            (TimeFrame.H4, TimeFrame.H12): 3,
            (TimeFrame.H4, TimeFrame.D1): 6,
            (TimeFrame.H4, TimeFrame.W1): 42,
            (TimeFrame.H12, TimeFrame.D1): 2,
            (TimeFrame.H12, TimeFrame.W1): 14,
            (TimeFrame.D1, TimeFrame.W1): 7,
        }
        
        ratio = conversion_ratios.get((source_tf, target_tf), 1)
        return max(1, source_bars // ratio)
    
    @staticmethod
    def validate_converted_data(
        original: pd.DataFrame,
        converted: pd.DataFrame,
        source_tf: TimeFrame,
        target_tf: TimeFrame
    ) -> Dict[str, bool]:
        """Validate converted data integrity."""
        results = {
            'count_reasonable': True,
            'no_gaps': True,
            'ohlc_valid': True,
            'volume_positive': True,
            'time_alignment': True
        }
        
        if original.empty or converted.empty:
            return results
        
        # Check if count is reasonable
        expected_count = TimeframeConverter.estimate_bars_count(
            source_tf, target_tf, len(original)
        )
        actual_count = len(converted)
        
        # Allow 10% variance
        if not (expected_count * 0.9 <= actual_count <= expected_count * 1.1):
            results['count_reasonable'] = False
        
        # Check for significant gaps
        if len(converted) > 1:
            time_diffs = converted.index.to_series().diff().dropna()
            target_freq_seconds = pd.Timedelta(TIMEFRAME_MAPPING[target_tf]).total_seconds()
            
            # Check if any gap is more than 2x the expected frequency
            max_gap_seconds = time_diffs.max().total_seconds()
            if max_gap_seconds > target_freq_seconds * 2:
                results['no_gaps'] = False
        
        # Validate OHLC relationships
        ohlc_valid = (
            (converted['low'] <= converted['high']).all() and
            (converted['low'] <= converted['open']).all() and
            (converted['low'] <= converted['close']).all() and
            (converted['high'] >= converted['open']).all() and
            (converted['high'] >= converted['close']).all()
        )
        results['ohlc_valid'] = ohlc_valid
        
        # Check volume is positive
        results['volume_positive'] = (converted['volume'] >= 0).all()
        
        # Check time alignment (should start on round intervals)
        if target_tf in [TimeFrame.H1, TimeFrame.H4, TimeFrame.H12, TimeFrame.D1]:
            # For hourly and daily timeframes, check alignment
            first_timestamp = converted.index[0]
            
            if target_tf == TimeFrame.H1:
                results['time_alignment'] = first_timestamp.minute == 0
            elif target_tf == TimeFrame.H4:
                results['time_alignment'] = first_timestamp.hour % 4 == 0 and first_timestamp.minute == 0
            elif target_tf == TimeFrame.H12:
                results['time_alignment'] = first_timestamp.hour % 12 == 0 and first_timestamp.minute == 0
            elif target_tf == TimeFrame.D1:
                results['time_alignment'] = (
                    first_timestamp.hour == 0 and 
                    first_timestamp.minute == 0
                )
        
        return results


def convert_data_for_backtesting(
    data: pd.DataFrame,
    source_tf: TimeFrame,
    target_tf: TimeFrame
) -> pd.DataFrame:
    """
    Convenience function to convert data for backtesting.
    Includes validation and error handling.
    """
    if source_tf == target_tf:
        return data.copy()
    
    converter = TimeframeConverter()
    
    # Convert the data
    converted = converter.convert_timeframe(data, source_tf, target_tf)
    
    # Validate conversion
    validation_results = converter.validate_converted_data(
        data, converted, source_tf, target_tf
    )
    
    # Log warnings for failed validations
    import logging
    logger = logging.getLogger(__name__)
    
    for check, passed in validation_results.items():
        if not passed:
            logger.warning(f"Data conversion validation failed: {check}")
    
    return converted


# Example usage functions
def create_multi_timeframe_dataset(
    base_data: pd.DataFrame,
    base_timeframe: TimeFrame,
    target_timeframes: List[TimeFrame]
) -> Dict[TimeFrame, pd.DataFrame]:
    """Create a multi-timeframe dataset from base data."""
    converter = TimeframeConverter()
    
    datasets = {base_timeframe: base_data.copy()}
    
    # Convert to each target timeframe
    for target_tf in target_timeframes:
        if target_tf != base_timeframe and converter.can_convert(base_timeframe, target_tf):
            datasets[target_tf] = converter.convert_timeframe(
                base_data, base_timeframe, target_tf
            )
    
    return datasets
