r"""QReal: a convenience wrapper around a computed q-real series.

The functional API in `arithmetic` and `truncated` is the stable core; this
class is sugar over it. A QReal holds a Laurent result (a valuation and a dense
coefficient list) together with a short label saying where it came from, and
offers the everyday read-outs plus operators that delegate straight back to the
functions:

    QReal("pi", 12).coeffs            -> [1, 1, 1, 0, ...]
    QReal("3/2", 12) + QReal("13/5")  -> q_add, a new QReal
    -QReal("sqrt(2)", 12)             -> q_neg (Jouteur), a Laurent QReal
    QReal("3/2", 12) * QReal("5/2")   -> q_mul, a new QReal

Operators carry the same caveats as the functions they call: + and * are the
series sum and product [x]_q +/* [y]_q, not [x +/* y]_q, and unary - is the
Jouteur PGL_2(Z) negation [-x]_q, not coefficient negation. See
docs/CORRECTNESS.md.
"""

from __future__ import annotations

from . import arithmetic
from .truncated import q_real_truncated

_DEFAULT_N = 12


class QReal:
    """A q-real [x]_q held to N coefficients, with read-outs and operators."""

    __slots__ = ("valuation", "coeffs", "label")

    def __init__(self, x: str, N: int = _DEFAULT_N):
        """Build [x]_q for real x >= 0, keeping its first N Taylor coefficients."""
        if N < 1:
            raise ValueError("N must be at least 1")
        self.valuation: int = 0
        self.coeffs: list[int] = q_real_truncated(str(x), N)
        self.label: str = f"[{x}]_q"

    # -- construction from a raw Laurent result (used by the operators) --------
    @classmethod
    def from_laurent(cls, valuation: int, coeffs: list[int], label: str) -> "QReal":
        """Wrap an already-computed (valuation, coeffs) Laurent result."""
        obj = cls.__new__(cls)
        obj.valuation = int(valuation)
        obj.coeffs = list(coeffs)
        obj.label = label
        return obj

    # -- read-outs -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.coeffs)

    @property
    def lowest_power(self) -> int:
        """The exponent of the first stored coefficient (the valuation)."""
        return self.valuation

    def radius_estimate(self) -> float:
        """Running-max root-test estimate of the radius of convergence.

        Defined only for a Taylor q-real (valuation 0); a negated/Laurent QReal
        has no power-series radius, so this raises. Delegates to
        `arithmetic.radius` via the coefficient list it already holds.
        """
        if self.valuation != 0:
            raise ValueError(
                "radius_estimate is for a Taylor q-real (valuation 0); this QReal "
                f"has valuation {self.valuation}"
            )
        if len(self.coeffs) < 2:
            raise ValueError("need at least 2 coefficients to estimate a slope")
        return _radius_from_coeffs(self.coeffs)

    def sign_pattern(self) -> str:
        """The sign of each coefficient as a string of '+', '-', '0'."""
        return " ".join("0" if c == 0 else ("+" if c > 0 else "-") for c in self.coeffs)

    def zero_run(self) -> tuple[int, int]:
        """The longest run of consecutive zero coefficients as (start, length).

        start is the index into `coeffs` (so the power is valuation + start);
        length is 0 when there is no zero coefficient. The first such longest run
        is reported on a tie.
        """
        best_start, best_len = 0, 0
        run_start, run_len = 0, 0
        for i, c in enumerate(self.coeffs):
            if c == 0:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
            else:
                run_len = 0
        return best_start, best_len

    # -- operators delegating to the functional API ----------------------------
    def __add__(self, other: "QReal") -> "QReal":
        x, y = _operand_real(self), _operand_real(other)
        n = min(len(self), len(other))
        return QReal.from_laurent(
            0, arithmetic.q_add(x, y, n), f"{self.label} + {other.label}"
        )

    def __mul__(self, other: "QReal") -> "QReal":
        x, y = _operand_real(self), _operand_real(other)
        n = min(len(self), len(other))
        return QReal.from_laurent(
            0, arithmetic.q_mul(x, y, n), f"{self.label} * {other.label}"
        )

    def __neg__(self) -> "QReal":
        x = _operand_real(self)
        v, c = arithmetic.q_neg(x, len(self))
        return QReal.from_laurent(v, c, f"[-{_strip(self.label)}]_q")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QReal):
            return NotImplemented
        return self.valuation == other.valuation and self.coeffs == other.coeffs

    def __repr__(self) -> str:
        return f"QReal(label={self.label!r}, valuation={self.valuation}, coeffs={self.coeffs})"


def _radius_from_coeffs(coeffs: list[int]) -> float:
    import math

    max_slope: float | None = None
    for k in range(1, len(coeffs)):
        c = coeffs[k]
        if c == 0:
            continue
        slope = math.log(abs(c)) / k
        if max_slope is None or slope > max_slope:
            max_slope = slope
    return math.inf if max_slope is None else math.exp(-max_slope)


def _strip(label: str) -> str:
    """Recover the x from a '[x]_q' label, else return the label unchanged."""
    if label.startswith("[") and label.endswith("]_q"):
        return label[1:-3]
    return label


def _operand_real(qr: "QReal") -> str:
    """The x string an operator should feed back to the functional API.

    Operators are defined for QReals that name a single real x (valuation 0,
    label '[x]_q'); a compound or negated QReal has no such x to recompute from.
    """
    x = _strip(qr.label)
    if qr.valuation != 0 or x == qr.label:
        raise ValueError(
            "QReal operators apply to a single-real q-real built as QReal(x, N); "
            f"this QReal is {qr.label!r}"
        )
    return x
