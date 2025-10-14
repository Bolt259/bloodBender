"""
Custom exceptions for the bloodBath sync client
"""


class TandemSyncError(Exception):
    """Base exception for sync errors"""
    pass


class RateLimitError(TandemSyncError):
    """Exception for rate limiting (429 responses)"""
    pass


class ChunkSizeError(TandemSyncError):
    """Exception when chunk size needs to be reduced (504 responses)"""
    pass


class BadRequestError(TandemSyncError):
    """Exception for bad API requests (400 responses)"""
    pass


class AuthenticationError(TandemSyncError):
    """Exception for authentication failures"""
    pass


class DataValidationError(TandemSyncError):
    """Exception for data validation errors"""
    pass


class APIConnectionError(TandemSyncError):
    """Exception for API connection problems"""
    pass
