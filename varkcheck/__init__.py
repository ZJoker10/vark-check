"""VARK Check — an agent that *tests* learning preferences instead of assuming them.

The package is deliberately dependency-light: retrieval, guardrails, memory and
the orchestrator use the Python standard library only, so the prototype runs
(and its tests run) on a bare interpreter. Streamlit is the only required
third-party package, and only for the user interface.
"""

__version__ = "1.0.0"
