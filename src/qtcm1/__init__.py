"""pyqtcm1: pure-Python reimplementation of the Neelin-Zeng QTCM1 (v2.3).

Phase 1 (foundations): constants, calendar, grid, FATD elliptic solver, and
the netCDF boundary-data reader. See the project scope document for the
migration plan and validation tiers.
"""

# Register the matplotlib converter for cftime axes (model output uses a
# noleap calendar).  xarray's .plot() imports nc_time_axis itself when the
# package is installed, but raw matplotlib calls (plt.plot(ds.time, ...))
# only work if the converter has been registered by an explicit import.
try:
    import nc_time_axis  # noqa: F401
except ImportError:      # minimal installs without the plotting extra
    pass

from . import constants
from .calendar import CalendarState, ModelCalendar, time_interp
from .grid import Grid

__version__ = '0.0.1'
__all__ = ['constants', 'CalendarState', 'ModelCalendar', 'time_interp',
           'Grid']
