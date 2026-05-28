"""Truncated Laurent series in q, with exact integer coefficients.

Every series is a pair (v, c) where v is the integer valuation and c is the
dense list of coefficients [c_v, c_{v+1}, ...]. A series is held only up to
some precision: coefficients of q^prec and higher are dropped. Coefficients
are exact Python ints, so nothing is lost to floating point.

This is the numeric kernel the MGO recursion runs on. The one operation that
is not elementary is series inversion, which uses Newton's iteration
y_{k+1} = y_k (2 - a y_k) and doubles the number of correct coefficients each
step. The fast path assumes a leading coefficient of +1 or -1, which is
always the case for the terms produced by the MGO formula; a Fraction-based
fallback covers the general case.
"""

from __future__ import annotations

from fractions import Fraction

Series = tuple[int, list[int]]


def trim(s: Series, prec: int) -> Series:
    v, c = s
    end = v + len(c)
    if end > prec:
        c = c[: max(prec - v, 0)]
    return v, c


def normalise(s: Series) -> Series:
    """Drop trailing zeros and advance the valuation past leading zeros."""
    v, c = s
    while c and c[-1] == 0:
        c.pop()
    if not c:
        return 0, []
    while c and c[0] == 0:
        c.pop(0)
        v += 1
    return v, c


def add(a: Series, b: Series, prec: int) -> Series:
    av, ac = a
    bv, bc = b
    v = min(av, bv)
    end = min(max(av + len(ac), bv + len(bc)), prec)
    out = [0] * (end - v)
    for i, x in enumerate(ac):
        if av + i < end:
            out[av + i - v] += x
    for i, x in enumerate(bc):
        if bv + i < end:
            out[bv + i - v] += x
    return normalise((v, out))


def mul(a: Series, b: Series, prec: int) -> Series:
    av, ac = a
    bv, bc = b
    if not ac or not bc:
        return 0, []
    v = av + bv
    max_len = max(prec - v, 0)
    out = [0] * max_len
    for i, ax in enumerate(ac):
        if ax == 0:
            continue
        max_j = max_len - i
        if max_j <= 0:
            break
        for j in range(min(len(bc), max_j)):
            out[i + j] += ax * bc[j]
    return normalise((v, out))


def scalar_mul(a: Series, s: int, prec: int) -> Series:
    v, c = a
    if s == 0:
        return 0, []
    return normalise((v, [x * s for x in c]))


def add_int(a: Series, n: int, prec: int) -> Series:
    return add(a, (0, [n] if n != 0 else []), prec)


def q_pow(k: int, prec: int) -> Series:
    """The monomial q^k, dropped to () if it sits at or beyond prec."""
    if k >= prec:
        return 0, []
    return k, [1]


def invert(a: Series, prec: int) -> Series:
    """Inverse of a series whose leading coefficient is +1 or -1.

    Writing a = q^v (sign + higher terms), the monic part u = sign * a / q^v
    is 1 + O(q), so Newton's iteration converges. The inverse of a is then
    sign * q^{-v} * u^{-1}. A leading coefficient other than +-1 routes to the
    Fraction fallback below.
    """
    v, c = a
    if not c:
        raise ZeroDivisionError("invert of zero series")
    leading = c[0]
    if leading not in (1, -1):
        return invert_general(a, prec)
    sign = leading
    monic = (0, [sign * x for x in c])
    target_len = max(prec - (-v), 1)
    y = (0, [1])
    cur_prec = 1
    while cur_prec < target_len:
        cur_prec = min(cur_prec * 2, target_len)
        uy = mul(monic, y, cur_prec)
        two_minus = add_int(scalar_mul(uy, -1, cur_prec), 2, cur_prec)
        y = mul(y, two_minus, cur_prec)
    yv, yc = y
    return trim(normalise((yv - v, [sign * x for x in yc])), prec)


def invert_general(a: Series, prec: int) -> Series:
    """Inverse for a non-unit leading coefficient, via Fraction arithmetic.

    Coefficients are cast back to int at the end; a non-integer result means
    the series was not invertible over the integers and raises.
    """
    v, c = a
    if not c:
        raise ZeroDivisionError("invert of zero series")
    target_len = max(prec - (-v), 1)
    leading = Fraction(c[0])
    monic = [Fraction(x) / leading for x in c]
    y = [Fraction(1)]
    cur = 1
    while cur < target_len:
        cur = min(cur * 2, target_len)
        uy = [Fraction(0)] * cur
        for i, mi in enumerate(monic):
            if i >= cur:
                break
            for j, yj in enumerate(y):
                if i + j >= cur:
                    break
                uy[i + j] += mi * yj
        two_minus = [-x for x in uy]
        two_minus[0] += 2
        new_y = [Fraction(0)] * cur
        for i, yi in enumerate(y):
            if i >= cur:
                break
            for j, tm in enumerate(two_minus):
                if i + j >= cur:
                    break
                new_y[i + j] += yi * tm
        y = new_y
    out_coeffs = []
    for fr in (yi / leading for yi in y):
        if fr.denominator != 1:
            raise ValueError("non-integer coefficient in series inverse")
        out_coeffs.append(int(fr.numerator))
    return trim(normalise((-v, out_coeffs)), prec)
