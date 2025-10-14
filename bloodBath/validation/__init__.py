"""
Validation framework for bloodBath system

Provides comprehensive end-to-end testing and validation capabilities
for the bloodBath pump data synchronization and processing pipeline.

Usage:
    from bloodBath.validation import ValidationTestFramework
    
    framework = ValidationTestFramework()
    success = framework.run_full_validation(years=5)
"""

#!/usr/bin/env python3
"""
Validation module for bloodBath data processing system.

Provides comprehensive validation capabilities for glucose, insulin, and 
pump data throughout the processing pipeline.
"""

from .test_framework import (
    ValidationTestFramework,
    ValidationMetrics,
    BatchMetadata
)

from .batch_retriever import (
    EnhancedBatchRetriever,
    BatchJob,
    BatchSchedule
)

from .data_organizer import (
    EnhancedDataOrganizer,
    ProcessingMetrics,
    ResamplingConfig
)

from .integrity_validator import (
    ComprehensiveValidator,
    ValidationResult,
    QualityMetrics,
    ValidationConfig
)

from .metrics_logger import (
    MetricsCollector,
    ValidationLogger,
    BatchJobMetrics,
    ValidationSessionMetrics,
    PipelineStageMetrics
)

__all__ = ['ValidationTestFramework', 'ValidationMetrics', 'BatchMetadata']