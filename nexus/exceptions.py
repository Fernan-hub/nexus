class NexusError(Exception):
    """Base exception for all Nexus-related errors."""
    pass

class ProcessingError(NexusError, RuntimeError):
    """Exception raised for errors during data processing."""
    pass

class MissingMetadataError(ProcessingError):
    """Exception raised when required metadata is missing."""
    pass
