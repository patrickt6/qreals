"""qreals: q-deformed rationals and reals via MGO continued fractions.

Public API:
    q_rational(p, s)        exact [p/s]_q as a rational function in q
    q_real_truncated(x, N)  first N stable Taylor coefficients of [x]_q
    q                       the sympy symbol q used by q_rational

the May-14 board lemmas in `expansions`:
    integer_part_prefix(x)              forced opening block [floor(x)]_q + 0*q^t
    coeffs_locked_by_convergent(cf, n)  (S_n, count) the n-th convergent pins down
    mgo_laurent(x, order)               [x]_q coefficients c_0..c_order
    shift_down(coeffs) / shift_up(...)  [x-1]_q and [x+1]_q shift relations

plus the coefficient read-outs in `coefficients`. Anything not re-exported
here is internal and may change.

arithmetic between q-reals in `arithmetic`, with a QReal wrapper:
    q_add(x, y, N)      first N coefficients of the series sum [x]_q + [y]_q
    q_mul(x, y, N)      first N coefficients of the series product [x]_q * [y]_q
    q_neg(x, N)         the Jouteur q-negation [-x]_q as (valuation, coeffs)
    negation_sum(x, N)  [x]_q + [-x]_q as (valuation, coeffs) (Ovsienko Ex. 6.4)
    finite_xnegx(x)     is [x]_q + [-x]_q a finite Laurent polynomial?
    radius(x, N)        running-max estimate of the radius of convergence of [x]_q
    QReal(x, N)         a convenience wrapper with the read-outs and operators

the independent bihomographic engine in `gosper`:
    q_gosper(x, y, op)  z([x]_q,[y]_q) as a rational function (op add or mul)
    gosper_coeffs(...)  its first N Taylor coefficients (the cross-check path)

the one-sided jump gap of a rational in `jumpgap`:
    jumpgap(p, s)   the two q-versions [p/s]_q^+ and [p/s]_q^-, their factored
                    gap, the exponent E, and the q-denominators S^+ and S^-
                    (Jouteur arXiv:2503.02122); JumpGap is the result type

The CF algorithm for [x]_q + [y]_q in
`q_sum` and the Jouteur negation in `negate`:
    negate(x, N)            the Jouteur [-x]_q for any real x (arXiv:2503.02122)
    transfer_matrix(cf)     the 2x2 MGO q-continuant block product
    q_sum_rational(x, y)    the CF algorithm for rational x, y > 0
    q_sum_irrational(x, y, N)  convergent iterator R_n/Q_n -> [x]_q + [y]_q
    finiteness_check(x, y, N)  is [x]_q + [y]_q a finite Laurent polynomial?

the q-arithmetic deficit in `deficit`:
    deficit(x, y, op, N)  the gap [x op y]_q - (series sum or product) for op
                          "+" or "*", with its q=1 and q=0 values and, for
                          rational inputs, the exact closed form; Deficit is the
                          result type
    negation_panel(x, N)  [x]_q + [-x]_q and the finite-or-infinite verdict
                          (Ovsienko Ex. 6.4); NegationPanel is the result type

quadratic-irrational arithmetic via continued-fraction transfer matrices in
`transfer` (the classical q=1 route of the golden+silver worked example):
    quad_arith(x, y, op)  exact closed form of x op y for quadratic irrationals
                          x, y and op in "add"/"sub"/"mul"/"div", read off from
                          the dominant eigenvector of K = M_x (x) M_y; QuadArith
                          is the result type (matrix, eigenvalues, value,
                          verified flag)
    combined_matrix(x, y) the 4x4 transfer matrix K = M_x (x) M_y
    periodic_cf(x)        (pre-period, period) of the CF of a quadratic irrational

the inline verification stamp in `verify`:
    verify(result)  run the cheap cross-checks for a computation result dict
    Stamp           the checks plus the one-line summary, writing nothing

two optional exploration helpers, each behind a light extra (no heavy ML in
core):
    featurize(x)    a named, fixed-length fingerprint of [x]_q for
                    nearest-neighbour over constants (`features`, numpy optional)
    oeis.lookup(c)  look a coefficient sequence up in the OEIS, re-verified
                    against the full b-file (`oeis`, needs requests)
"""

from .arithmetic import (
    finite_xnegx,
    negation_sum,
    q_add,
    q_mul,
    q_neg,
    radius,
)
from .coefficients import (
    coefficient_max_abs,
    first_negative_coefficient_index,
    first_nonzero_coefficient_index,
    number_of_zeros,
)
from .deficit import Deficit, NegationPanel, deficit, negation_panel
from .factor import (
    QRealFactor,
    SProperties,
    classify_poles,
    degree_collapse,
    denominator_expr,
    factor_qreal,
    numerator_expr,
    s_atlas,
    s_properties,
    s_regime,
    saturation_explorer,
)
from .expansions import (
    coeffs_locked_by_convergent,
    format_laurent,
    integer_part_prefix,
    mgo_laurent,
    shift_down,
    shift_up,
)
from .features import Fingerprint, feature_distance, feature_names, featurize, nearest
from .gosper import gosper_coeffs, q_gosper
from .jumpgap import JumpGap, jumpgap
from .negate import negate
from .q_sum import (
    Approximant,
    FinitenessReport,
    QSumIrrational,
    QSumRational,
    finiteness_check,
    q_sum_irrational,
    q_sum_rational,
    transfer_matrix,
)
from .qint_factor import (
    QIntFactor,
    canonical_multiset,
    qint_factor,
    qint_factor_peeling,
    qint_product,
)
from .qreal import QReal
from .rational import q, q_int, q_int_qinv, q_rational
from .store import SavedEntry, SavedStore, user_data_dir
from .transfer import (
    QuadArith,
    combined_matrix,
    continuant_matrix,
    periodic_cf,
    quad_arith,
)
from .truncated import q_real_truncated
from .verify import Stamp, verify

# Imported as a submodule, not a name, so `import qreals` exposes
# `qreals.oeis.lookup(...)` without pulling in the optional requests dependency
# at import time (oeis imports requests lazily, only when a lookup runs).
from . import oeis  # noqa: E402

# The export renderers (JSON, CSV, LaTeX table, Magma) for a saved entry or a
# list. Pure sympy plus the standard library; platformdirs is preferred for the
# store location but optional. The qprov link lives in `qreals.provenance` and
# is a lazy, optional import, so the core never pulls qprov in.
from . import exports  # noqa: E402

__version__ = "0.1.3"

__all__ = [
    "q_rational",
    "q_real_truncated",
    "q_add",
    "q_mul",
    "q_neg",
    "negate",
    "q_sum_rational",
    "q_sum_irrational",
    "transfer_matrix",
    "quad_arith",
    "QuadArith",
    "periodic_cf",
    "continuant_matrix",
    "combined_matrix",
    "finiteness_check",
    "QSumRational",
    "QSumIrrational",
    "Approximant",
    "FinitenessReport",
    "negation_sum",
    "finite_xnegx",
    "radius",
    "QReal",
    "q_gosper",
    "gosper_coeffs",
    "jumpgap",
    "JumpGap",
    "deficit",
    "Deficit",
    "negation_panel",
    "NegationPanel",
    "qint_factor",
    "qint_factor_peeling",
    "QIntFactor",
    "qint_product",
    "canonical_multiset",
    "factor_qreal",
    "QRealFactor",
    "s_properties",
    "SProperties",
    "s_regime",
    "s_atlas",
    "saturation_explorer",
    "degree_collapse",
    "classify_poles",
    "numerator_expr",
    "denominator_expr",
    "q",
    "q_int",
    "q_int_qinv",
    "first_nonzero_coefficient_index",
    "first_negative_coefficient_index",
    "coefficient_max_abs",
    "number_of_zeros",
    "integer_part_prefix",
    "coeffs_locked_by_convergent",
    "mgo_laurent",
    "shift_down",
    "shift_up",
    "format_laurent",
    "verify",
    "Stamp",
    "featurize",
    "feature_names",
    "feature_distance",
    "nearest",
    "Fingerprint",
    "oeis",
    "SavedStore",
    "SavedEntry",
    "user_data_dir",
    "exports",
    "__version__",
]
