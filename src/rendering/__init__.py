"""Offline inspection of recorded simulator state; no sensor rendering."""

from driversi._compat import extend_path as _extend_legacy_path
_extend_legacy_path(__name__, __path__)

from driversi.rendering.bev.replay import export_replay

__all__ = ["export_replay"]
