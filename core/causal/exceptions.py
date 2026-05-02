"""SCM-specific runtime errors."""


class SimplePathEnumerationCap(RuntimeError):
    """Too many simple paths between two nodes or hit explicit path budget."""
