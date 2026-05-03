## Coding Rules

1. Code should be compact and readable
   - Ideally everything is small methods on a class, no loose functions
   - There should only be one class per file
   - Use inheritence and composition to keep individual classes small and focussed
   - Logically grouped code (especially blocks) should be separated by a blank newline

2. Code should be reusable
   - Don't keep writing the same kind of calculations, algorithms, or logic across many files
   - Instead, create modules, classes, and other reusable structures with named wrappers

3. Never use fallbacks
   - Never use silent fallbacks, just throw an exception when something isn't as it should be
   - Never hide issues, expose the flaws of the system so we can fix them

4. No optional or alternative paths, and no magic or tunable values
   - There is only one system, and that is the full system with everything wired in
   - Use dynamic/derived values in favor of tunable parameters

### The Shape of a Module

```
- __init__.py
- base.py
- builder.py
- specialist1.py
- specialist2.py
- ...
```

For example (from another project):

__init__.py

```python
"""Topology data structures

The topology subsystem defines *runtime* topology structures. It does not
perform compilation/lowering of manifest DSL. Compilation concerns belong to
the manifest compiler.
"""

from __future__ import annotations

from caramba.framework.topology.base import BaseTopology
from caramba.framework.topology.builder import TopologyBuilder
from caramba.framework.topology.graph import GraphTopology

__all__ = [
    "BaseTopology",
    "TopologyBuilder",
    "GraphTopology",
]
```

base.py

```python
"""Base interface for topology

Topologies determine the computational graph of the model.
They lay out operations in a specific pattern (sequential, residual, etc).
"""

from __future__ import annotations

from abc import ABC
from typing import Any


class BaseTopology(ABC):
    """Abstract base for topology implementations.
    
    Topologies define how operations are connected. Different topology
    types (Graph, Residual, Repeated) provide different connection patterns.
    """

    def __init__(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        raise NotImplementedError("Subclasses must implement to_dict()")
```

builder.py

```python
"""Topology builder for caramba

Builds topology structures from configuration.
"""

from __future__ import annotations

from typing import Any

from .base import BaseTopology
from .graph import GraphTopology


class TopologyBuilder:
    """Topology builder for caramba (legacy interface)."""
    
    def __init__(self, *, config: dict[str, Any]) -> None:
        self.config = config

    def build(self) -> BaseTopology:
      """Build a topology from configuration.
      
      Takes a topology configuration dictionary and returns
      the appropriate topology instance.
      """
      topo_type = config.get("type", "GraphTopology")
      
      match topo_type:
         case "GraphTopology":
               return GraphTopology(
                  inputs=config.get("inputs", []),
                  nodes=config.get("nodes", []),
                  outputs=config.get("outputs", []),
               )
         case _:
               raise ValueError(f"Unrecognized topology type: {topo_type}")
```

graph.py

```python
"""Graph topology for caramba

A GraphTopology represents a directed acyclic graph (DAG) of operations.
Nodes are operations connected by named edges (tensor references).
"""

from __future__ import annotations

from typing import Any

from .base import BaseTopology


class GraphTopology(BaseTopology):
    """Graph topology representing a DAG of operations.
    
    A graph topology consists of:
    - inputs: Named input tensors
    - nodes: Operations with typed edges
    - outputs: Named output tensors
    """
    
    def __init__(
        self,
        *,
        inputs: list[str] | None = None,
        nodes: list[dict[str, Any]] | None = None,
        outputs: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.inputs: list[str] = inputs or []
        self.nodes: list[dict[str, Any]] = nodes or []
        self.outputs: list[str] = outputs or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "GraphTopology",
            "inputs": self.inputs,
            "nodes": self.nodes,
            "outputs": self.outputs,
        }
```
