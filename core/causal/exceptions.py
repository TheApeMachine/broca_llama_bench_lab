"""SCM-specific runtime errors."""


class SimplePathEnumerationCap(RuntimeError):
    """Raised when simple-path enumeration exceeds an explicit path budget (optional legacy / strict modes)."""

    def __init__(
        self,
        message: str,
        *,
        source_node: str | None = None,
        target_node: str | None = None,
        cap: int | None = None,
        path_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.source_node = source_node
        self.target_node = target_node
        self.cap = cap
        self.path_count = path_count

    def __str__(self) -> str:
        base = super().__str__()
        meta: list[str] = []
        if self.source_node is not None:
            meta.append(f"source_node={self.source_node!r}")
        if self.target_node is not None:
            meta.append(f"target_node={self.target_node!r}")
        if self.cap is not None:
            meta.append(f"cap={self.cap}")
        if self.path_count is not None:
            meta.append(f"path_count={self.path_count}")
        if meta:
            return f"{base} ({', '.join(meta)})"
        return base
