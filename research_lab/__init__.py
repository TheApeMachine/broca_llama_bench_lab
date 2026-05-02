"""Research and publication scaffolding for the mosaic substrate.

Sibling package to ``core``. Imports flow one direction only:
``research_lab`` may consume the substrate; ``core`` must never import from
``research_lab``. The CLI in ``core.main`` lazy-imports a few entry points
here as a convenience, but the substrate runtime does not depend on this
package at module-load time.
"""
