from typing import Protocol


class Frontend(Protocol):
    """UI or shell entry surface for running the Mosaic control plane interactively.

    Implementations own how the process blocks (or yields) and how errors reach
    the operator; callers treat :meth:`run` as the primary lifecycle hook until
    the front end exits normally or raises.
    """

    def run(self) -> None:
        """Start the front end; expected to block until shutdown.

        Implementations may perform setup before entering their main loop. Unless
        documented otherwise, errors propagate to the caller (this protocol does
        not require swallowing exceptions).
        """
        ...

