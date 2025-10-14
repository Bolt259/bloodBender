"""
CSV Data Quality Validator

Validates pump CSV data for synthetic/placeholder patterns before ingestion.
Detects low variance, repeated values, and unrealistic BG patterns that 
indicate fake data that should not be used for training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results from CSV validation analysis"""
    is_valid: bool
    confidence_score: float  # 0.0 = definitely synthetic, 1.0 = definitely real
    issues: List[str]
    stats: Dict[str, Any]
    
    def __str__(self):
        status = "VALID" if self.is_valid else "SYNTHETIC"
        return f"{status} (confidence: {self.confidence_score:.3f}) - {len(self.issues)} issues"

class CSVValidator:
    """
    Validates CSV pump data for synthetic patterns.
    
    Key detection criteria:
    - Excessive repetition of exact values (esp. 100.0 mg/dL)
    - Low variance in BG readings
    - Unrealistic BG patterns
    - Statistical anomalies indicating placeholder data
    """
    
    def __init__(self, 
                 max_dominant_percentage: float = 25.0,
                 max_narrow_range_percentage: float = 40.0,
                 min_variance_threshold: float = 200.0,
                 min_unique_values: int = 50):
        """
        Initialize validator with detection thresholds.
        
        Args:
            max_dominant_percentage: Max % of records that can be the same exact value
            max_narrow_range_percentage: Max % in narrow range (95-105 mg/dL) 
            min_variance_threshold: Minimum BG variance for realistic data
            min_unique_values: Minimum unique BG values expected
        """
        self.max_dominant_percentage = max_dominant_percentage
        self.max_narrow_range_percentage = max_narrow_range_percentage  
        self.min_variance_threshold = min_variance_threshold
        self.min_unique_values = min_unique_values
        
    def validate_csv(self, csv_path: str) -> ValidationResult:
        """
        Validate a CSV file for synthetic data patterns.
        
        Args:
            csv_path: Path to CSV file to validate
            
        Returns:
            ValidationResult with analysis and recommendations
        """
        try:
            # Load CSV with comment handling
            df = pd.read_csv(csv_path, comment='#', index_col=False)
            
            if len(df) == 0:
                return ValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    issues=["Empty CSV file"],
                    stats={"records": 0}
                )
                
            # Ensure we have bg column
            if 'bg' not in df.columns:
                return ValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    issues=["Missing 'bg' column"],
                    stats={"records": len(df), "columns": list(df.columns)}
                )
                
            return self._analyze_bg_data(df)
            
        except Exception as e:
            logger.error(f"Error validating CSV {csv_path}: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[f"Failed to read CSV: {e}"],
                stats={}
            )
    
    def _analyze_bg_data(self, df: pd.DataFrame) -> ValidationResult:
        """Analyze BG data for synthetic patterns"""
        bg_series = df['bg']
        issues = []
        stats = {}
        
        # Basic statistics
        records = len(df)
        bg_mean = bg_series.mean()
        bg_std = bg_series.std()
        bg_var = bg_series.var()
        bg_min = bg_series.min()
        bg_max = bg_series.max()
        unique_values = bg_series.nunique()
        
        stats.update({
            'records': records,
            'bg_mean': bg_mean,
            'bg_std': bg_std,
            'bg_variance': bg_var,
            'bg_range': (bg_min, bg_max),
            'unique_values': unique_values
        })
        
        # Test 1: Dominant value detection (especially 100.0)
        value_counts = bg_series.value_counts()
        most_frequent_val = value_counts.index[0]
        most_frequent_count = value_counts.iloc[0]
        dominant_percentage = (most_frequent_count / records) * 100
        
        stats.update({
            'most_frequent_value': most_frequent_val,
            'most_frequent_count': most_frequent_count,
            'dominant_percentage': dominant_percentage
        })
        
        if dominant_percentage > self.max_dominant_percentage:
            issues.append(f"Dominant value {most_frequent_val} appears {dominant_percentage:.1f}% of time (>{self.max_dominant_percentage}%)")
            
        # Test 2: Narrow range concentration (95-105 mg/dL)
        narrow_range_mask = (bg_series >= 95) & (bg_series <= 105)
        narrow_range_count = narrow_range_mask.sum()
        narrow_range_percentage = (narrow_range_count / records) * 100
        
        stats['narrow_range_percentage'] = narrow_range_percentage
        
        if narrow_range_percentage > self.max_narrow_range_percentage:
            issues.append(f"Too many values in 95-105 range: {narrow_range_percentage:.1f}% (>{self.max_narrow_range_percentage}%)")
            
        # Test 3: Low variance detection
        try:
            bg_var_float = np.asarray(bg_var).item()
            if bg_var_float < self.min_variance_threshold:
                issues.append(f"Low BG variance: {bg_var_float:.1f} (minimum: {self.min_variance_threshold})")
        except (ValueError, TypeError, OverflowError):
            issues.append("Invalid variance calculation")
            
        # Test 4: Insufficient unique values
        if unique_values < self.min_unique_values:
            issues.append(f"Too few unique values: {unique_values} (minimum: {self.min_unique_values})")
            
        # Test 5: Unrealistic BG ranges
        if bg_min < 20 or bg_max > 600:
            issues.append(f"Unrealistic BG range: {bg_min}-{bg_max} mg/dL")
            
        # Test 6: Suspicious exact value 100.0 concentration
        exact_100_count = (bg_series == 100.0).sum()
        exact_100_percentage = (exact_100_count / records) * 100
        
        stats['exact_100_percentage'] = exact_100_percentage
        
        if exact_100_percentage > 20.0:  # More than 20% exactly 100.0 is suspicious
            issues.append(f"Suspicious concentration of exact 100.0 values: {exact_100_percentage:.1f}%")
            
        # Calculate confidence score
        confidence_score = self._calculate_confidence(stats, issues)
        
        # Determine validity
        is_valid = confidence_score >= 0.5 and len(issues) <= 2
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues=issues,
            stats=stats
        )
    
    def _calculate_confidence(self, stats: Dict, issues: List[str]) -> float:
        """
        Calculate confidence score (0.0 = synthetic, 1.0 = real data)
        
        Uses multiple factors:
        - Number of issues (more issues = lower confidence)
        - Dominant value percentage 
        - Variance level
        - Unique value diversity
        """
        # Start with base confidence
        confidence = 1.0
        
        # Penalize for each issue
        confidence -= len(issues) * 0.15
        
        # Penalize heavily for dominant values
        dominant_pct = stats.get('dominant_percentage', 0)
        if dominant_pct > 30:
            confidence -= 0.4
        elif dominant_pct > 20:
            confidence -= 0.2
            
        # Penalize for low variance  
        variance = stats.get('bg_variance', 0)
        try:
            variance_float = float(variance) if variance is not None else 0
            if variance_float < 100:
                confidence -= 0.3
            elif variance_float < 200:
                confidence -= 0.15
        except (ValueError, TypeError):
            confidence -= 0.2  # Penalize for invalid variance
            
        # Penalize for narrow range concentration
        narrow_pct = stats.get('narrow_range_percentage', 0)
        if narrow_pct > 50:
            confidence -= 0.25
        elif narrow_pct > 40:
            confidence -= 0.1
            
        # Penalize for low unique value count
        unique_ratio = stats.get('unique_values', 0) / max(stats.get('records', 1), 1)
        if unique_ratio < 0.02:  # Less than 2% unique values
            confidence -= 0.2
            
        # Bonus for realistic variance and diversity
        try:
            variance_float = float(variance) if variance is not None else 0
            if variance_float > 500 and unique_ratio > 0.05:
                confidence += 0.1
        except (ValueError, TypeError):
            pass  # Skip bonus if variance is invalid
            
        return max(0.0, min(1.0, confidence))
    
    def validate_directory(self, directory_path: str, 
                          pattern: str = "*.csv") -> Dict[str, ValidationResult]:
        """
        Validate all CSV files in a directory.
        
        Args:
            directory_path: Path to directory containing CSV files
            pattern: File pattern to match (default: "*.csv")
            
        Returns:
            Dictionary mapping filename to ValidationResult
        """
        directory = Path(directory_path)
        results = {}
        
        for csv_file in directory.glob(pattern):
            if csv_file.is_file():
                result = self.validate_csv(str(csv_file))
                results[csv_file.name] = result
                
        return results
    
    def generate_report(self, results: Dict[str, ValidationResult]) -> str:
        """Generate a summary report of validation results"""
        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.is_valid)
        synthetic_files = total_files - valid_files
        
        report = []
        report.append("CSV VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Total files analyzed: {total_files}")
        report.append(f"Valid files: {valid_files}")
        report.append(f"Synthetic/Invalid files: {synthetic_files}")
        report.append("")
        
        # List synthetic files
        if synthetic_files > 0:
            report.append("SYNTHETIC/INVALID FILES:")
            report.append("-" * 30)
            for filename, result in results.items():
                if not result.is_valid:
                    report.append(f"{filename}: {result}")
                    for issue in result.issues:
                        report.append(f"  - {issue}")
                    report.append("")
        
        # Summary of valid files
        if valid_files > 0:
            report.append("VALID FILES SUMMARY:")
            report.append("-" * 30)
            valid_results = [r for r in results.values() if r.is_valid]
            avg_confidence = sum(r.confidence_score for r in valid_results) / len(valid_results)
            report.append(f"Average confidence: {avg_confidence:.3f}")
            
        return "\n".join(report)

# Convenience functions
def validate_single_csv(csv_path: str) -> ValidationResult:
    """Quick validation of a single CSV file"""
    validator = CSVValidator()
    return validator.validate_csv(csv_path)

def validate_pump_directory(pump_dir: str) -> Dict[str, ValidationResult]:
    """Validate all CSV files for a specific pump"""
    validator = CSVValidator()
    return validator.validate_directory(pump_dir)

def is_synthetic_data(csv_path: str) -> bool:
    """Simple boolean check if CSV contains synthetic data"""
    result = validate_single_csv(csv_path)
    return not result.is_valid