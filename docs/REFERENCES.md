# Certificate references

Every source a `qreals` certificate cites is hard-coded in the registry
`src/qreals/refs.py` (the dict `REFERENCES`). A certificate never fetches at
runtime: each in-text citation renders as a hyperlink to the URL recorded here,
and each certificate ends with a hyperlinked Sources list.

This note is the checked-in record of those URLs. Every URL below was checked to
resolve on 2026-05-24. If a link later breaks, fix the URL in `refs.py` and
update the matching line here. `tests/test_refs.py` asserts that the registry and
this note name the same URLs, so the two cannot drift apart unnoticed.

## Registry keys and URLs

### `mgo-rat`

S. Morier-Genoud and V. Ovsienko, "q-deformed rationals and q-continued
fractions", Forum Math. Sigma 8 (2020), e13.

- arXiv: https://arxiv.org/abs/1812.00170
- DOI (Cambridge, Forum Math. Sigma): https://doi.org/10.1017/fms.2020.9

Cited in certificates as "MGO eqn 1.1" (the construction folded in section (b)).

### `mgo-real`

S. Morier-Genoud and V. Ovsienko, "On q-deformed real numbers", Exp. Math. 31
(2022), no. 2, 652-660.

- arXiv: https://arxiv.org/abs/1908.04365
- DOI (Taylor and Francis, Experimental Mathematics): https://doi.org/10.1080/10586458.2019.1671922

Cited in certificates as "MGO Proposition 1.1" (the coefficient-stabilisation
bound used by the `[x]_q` series certificates). Per `docs/CORRECTNESS.md`, this
Proposition 1.1 is in this paper, not in the Forum Math. Sigma paper.

### `mgo-survey`

S. Morier-Genoud and V. Ovsienko, "q-deformed rationals and irrationals" (a
survey written for the second edition of the Mathematical Omnibus).

- arXiv: https://arxiv.org/abs/2503.23834

Cited in certificates as the accessible overview at the head of the Sources list.

## How resolution was checked

- The three arXiv abstract pages were fetched and returned the paper page with
  the matching title and authors (Morier-Genoud and Ovsienko).
- The two DOIs were resolved through https://doi.org and returned a 302 redirect
  to the publisher landing page (Cambridge Core for `mgo-rat`, Taylor and Francis
  for `mgo-real`), which is the expected resolve behaviour for a live DOI.
