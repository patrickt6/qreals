import pytest

from qreals import (
    coeffs_locked_by_convergent,
    format_laurent,
    integer_part_prefix,
    mgo_laurent,
    q_real_truncated,
    shift_down,
    shift_up,
)


# --- note 3: the MGO Laurent read-out -------------------------------------


def test_22_over_7_matches_board_example():
    # [22/7]_q = 1 + q + q^2 + q^9 + O(q^10)  (may14 note 3).
    assert mgo_laurent("22/7", 9) == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]


def test_mgo_laurent_agrees_with_truncated():
    # mgo_laurent(x, order) is q_real_truncated(x, order + 1).
    assert mgo_laurent("pi", 12) == q_real_truncated("pi", 13)


def test_mgo_laurent_order_zero_is_constant_term():
    assert mgo_laurent("pi", 0) == [1]
    assert mgo_laurent("1/2", 0) == [0]  # [1/2]_q has valuation 1


def test_mgo_laurent_negative_order_raises():
    with pytest.raises(ValueError):
        mgo_laurent("pi", -1)


# --- note 1: the integer-part prefix --------------------------------------


@pytest.mark.parametrize(
    "x,expected",
    [
        ("22/7", [1, 1, 1, 0]),  # floor 3
        ("5/2", [1, 1, 0]),  # floor 2
        ("7", [1, 1, 1, 1, 1, 1, 1, 0]),  # floor 7
        ("3/2", [1, 0]),  # floor 1
        ("1/2", [0]),  # floor 0
        ("pi", [1, 1, 1, 0]),  # floor 3
    ],
)
def test_integer_part_prefix(x, expected):
    assert integer_part_prefix(x) == expected


def test_prefix_matches_leading_laurent_coefficients():
    # the forced prefix really is the start of the full expansion.
    for x in ("22/7", "5/2", "pi", "12/5"):
        prefix = integer_part_prefix(x)
        full = mgo_laurent(x, len(prefix) + 4)
        assert full[: len(prefix)] == prefix


def test_prefix_negative_floor_raises():
    with pytest.raises(ValueError):
        integer_part_prefix("-1/2")


# --- note 2: how many coefficients a convergent pins down ------------------


def test_coeffs_locked_pi_second_convergent():
    # pi = [3, 7, 15, 1, 292, ...]; S_2 = 10, count = S_2 - 1 = 9.
    s_n, count = coeffs_locked_by_convergent([3, 7, 15, 1, 292], 2)
    assert s_n == 10
    assert count == 9


def test_coeffs_locked_count_is_tight_for_22_over_7():
    # The convergent [3,7] locks count = 9 coefficients (c_0..c_8). The 9th
    # index (q^9) is the first that may differ, and it does: [22/7]_q has 1
    # there while [pi]_q has 0.
    _, count = coeffs_locked_by_convergent([3, 7, 15], 2)
    pi = q_real_truncated("pi", count + 5)
    conv = q_real_truncated("22/7", count + 5)
    assert pi[:count] == conv[:count]  # agree on the locked block
    assert pi[count] != conv[count]  # diverge at the first free power


def test_coeffs_locked_validates_arguments():
    with pytest.raises(ValueError):
        coeffs_locked_by_convergent([3, 7], 0)
    with pytest.raises(ValueError):
        coeffs_locked_by_convergent([3, 7], 5)


# --- note 5: the shift relations ------------------------------------------


def test_shift_up_is_q_times_plus_one():
    # [x+1]_q = q[x]_q + 1: prepend a 1, push everything up a power.
    assert shift_up([1, 1, 1]) == [1, 1, 1, 1]
    assert shift_up([1, 0, -1]) == [1, 1, 0, -1]


def test_shift_down_inverts_shift_up():
    c = [1, 0, 0, 1, 0, -1]
    assert shift_down(shift_up(c)) == c


def test_shift_down_requires_unit_constant_term():
    with pytest.raises(ValueError):
        shift_down([0, 1, 1])  # constant term not 1


def test_shift_down_twice_on_pi_lands_in_open_unit_interval_one_two():
    # pi - 2 ~ 1.1416 lies in (1, 2): floor 1, so [pi-2]_q opens 1 + 0*q.
    coeffs = mgo_laurent("pi", 12)
    pim2 = shift_down(shift_down(coeffs))
    assert pim2[0] == 1 and pim2[1] == 0
    # the floor-1 prefix is exactly this opening.
    assert pim2[:2] == integer_part_prefix("pi-2")


def test_shift_chain_matches_direct_pi_minus_two():
    coeffs = mgo_laurent("pi", 13)
    pim2_shift = shift_down(shift_down(coeffs))
    pim2_direct = q_real_truncated("pi-2", len(pim2_shift))
    assert pim2_shift == pim2_direct


def test_pi_minus_two_zero_run_ends_at_a7():
    # Resolution of the note-5 tension: [pi]_q has a 0 at q^9, so the leading
    # zero run of [pi-2]_q runs a_2 = ... = a_7 = 0 and first reappears at a_8.
    pi = mgo_laurent("pi", 13)
    assert pi[9] == 0
    pim2 = shift_down(shift_down(pi))
    assert all(pim2[k] == 0 for k in range(2, 8))  # a_2 .. a_7 vanish
    assert pim2[8] == 1  # a_8 is the first nonzero


# --- formatting -----------------------------------------------------------


def test_format_laurent_reads_like_the_board():
    assert (
        format_laurent([1, 1, 1, 0, 0, 0, 0, 0, 0, 1]) == "1 + q + q^2 + q^9 + O(q^10)"
    )
    assert format_laurent([1, 0, -1]) == "1 - q^2 + O(q^3)"
    assert format_laurent([0]) == "0 + O(q^1)"
