class RFunctionNotFoundError(Exception):
    """Raised when an R function cannot be found or loaded."""
    pass

class RPackageNotLoadedError(Exception):
    """Raised when an R package fails to load."""
    pass