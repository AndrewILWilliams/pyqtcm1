"""Cloud-fraction scheme: port of ``cloud`` (clrad.F90, v2.3 form).

Four radiatively active cloud types over clear sky (type 0):
type 1 deep+CsCc (proportional to convective precipitation, capped at 1),
type 2 cirrus (1.5 x type 1 before overlap), type 3 stratus and type 4
AsAc+CuSc (constant reference amounts). Random-overlap bookkeeping gives
non-overlapped covers summing to <= 1; clear sky is the residual
(NZ eq. 4.37 context; Chou 1997).

The ``OBSCLD`` option (ISCCP type-3 climatology from data) is accepted via
``cldtot3`` override; the reader lives with the boundary data (P4 wiring).
"""

from __future__ import annotations

import numpy as np

#: reference (mean) cloud cover per type 1..4 (clrad.F90 ``cldref``)
CLDREF = np.array([0.1051000, 0.1047000, 0.1096000, 0.2234000])
CL1P = 7.76275869e-4      #: cloud-1 fraction per unit Qc [m2/W]
CL2FAC = 1.5              #: cirrus/deep ratio before overlap


def cloud(Qc: np.ndarray, *, cldtot3: np.ndarray | None = None) -> dict:
    """Cloud covers from convective heating Qc [W m-2].

    Returns ``cld`` of shape (5, ny, nx) (index 0 = clear sky) and ``cl1``
    (= cld[1], the deep-cloud output field).
    """
    if cldtot3 is None:                       # default: constant stratus
        cldtot3 = np.full_like(Qc, CLDREF[2] / (1.0 - CLDREF[0] - CLDREF[1]))
    cldtot4 = np.minimum(CLDREF[3] / (1.0 - CLDREF[0] - CLDREF[1]),
                         1.0 - cldtot3)

    cld1 = np.minimum(CL1P * Qc, 1.0)
    cldtot2 = np.minimum(cld1 * CL2FAC, 1.0)
    cld2 = cldtot2 * (1.0 - cld1)
    cld3 = cldtot3 * (1.0 - cld1 - cld2)
    cld4 = cldtot4 * (1.0 - cld1 - cld2)
    cld0 = 1.0 - cld1 - cld2 - cld3 - cld4
    cld = np.stack([cld0, cld1, cld2, cld3, cld4])
    return dict(cld=cld, cl1=cld1)
