#!/usr/bin/env python3
"""Free-running cold-started control run with monthly-mean output.

Usage: python run_control10.py [nyears] [outfile]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qtcm1.driver import ControlRun

nyears = int(sys.argv[1]) if len(sys.argv) > 1 else 10
out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    '~/work/run/py_ctrl_monthly.npz')

run = ControlRun(os.path.expanduser('~/work/data/qtcm1_bnd_r64x42'))
t0 = time.time()


def progress(dayofmodel, date):
    yr = dayofmodel // 365
    el = time.time() - t0
    print(f'year {yr:2d} done  ({el:.0f} s elapsed, '
          f'{el / dayofmodel * 365:.0f} s/yr)', flush=True)


run.run_years(nyears, progress=progress)
run.save_monthly(out)
print(f'wrote {out} ({len(run.monthly)} months)')
