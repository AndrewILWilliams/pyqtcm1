"""pyqtcm1: pure-Python reimplementation of the Neelin-Zeng QTCM1 (v2.3).

Phase 1 (foundations): constants, calendar, grid, FATD elliptic solver, and
the netCDF boundary-data reader. See the project scope document for the
migration plan and validation tiers.
"""

from . import constants
from .calendar import CalendarState, ModelCalendar, time_interp
from .grid import Grid

__version__ = '0.0.1'
__all__ = ['constants', 'CalendarState', 'ModelCalendar', 'time_interp',
           'Grid']
