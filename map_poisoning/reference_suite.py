"""Compatibility entry point for the autonomous reference-figure suite."""
from .reference_experiments import main, run_reference_suite

__all__ = ["main", "run_reference_suite"]


if __name__ == "__main__":
    raise SystemExit(main())
