#!/usr/bin/env python3
"""
scorer.py 2.9.0: clamp the Q1 query-time charge to the DB's actual charge
buckets before building the L0 key.

Diagnostic finding (this session, from the user's own "should resolve at
level 0" notebook trace on co_nh3_6_octahedral): the live constraints.json
only ever contains charge in {-1, 0, 1} for Q1 (verified: every metal,
every arity, no exceptions -- key_builder.py's q1_bond_lengths() builds
the raw fit-time key the same unclamped way, so this is either a genuine
data-coverage gap for |charge|>1 complexes or an upstream bucketing step;
either way the DELIVERED database only has these three buckets). The
query-time key in this file used mol.charge raw and unclamped, so any
complex with |charge|>1 (e.g. [Co(NH3)6]3+, charge=+3; also common cases
like [Fe(CN)6]3-, [Cr(H2O)6]3+) can never match ANY Q1 candidate that still
carries the charge field, including every intermediate donor-subclass-
coarsening level (L2-L4, which all still condition on charge) -- the
resolver has to skip straight past donor-subclass and geometry
specificity together, landing on the fully generic (metal, donor_family)
2-tuple. Confirmed via the notebook trace: co_nh3_6_octahedral resolved at
level 6/10 (Co, N_donor) rather than anywhere near level 0-2, despite
[Co(NH3)6]3+ being an extremely well-represented case in the CSD.

Fix: clamp charge to the nearest of the DB's known buckets before building
the key, so a high-charge complex still benefits from all the donor-
subclass/geometry-specific data available for the SAME donor chemistry at
a nearby, in-DB charge state, instead of discarding donor subclass and
geometry together. Not a full fix for the underlying data-coverage gap
(a genuinely different charge state COULD have systematically different
bond lengths), but strictly better than the current total collapse to a
donor-family-blind fallback, and the deepest fallback level (metal,
donor_family) was already charge-blind anyway.
"""
import sys

PATH = "src/jacobik/ik/scorer.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


# ---------------------------------------------------------------------------
# C0: version-history docstring
# ---------------------------------------------------------------------------
OLD_TAIL = '''2.8.0 : _score_components/score/score_breakdown gain an optional'''
NEW_TAIL = '''2.9.0 : Q1 key construction clamps mol.charge to the DB's actual charge
         buckets, {-1, 0, 1} (verified: every metal, every Q1 arity in the
         live constraints.json -- confirmed via a live notebook trace
         showing co_nh3_6_octahedral, charge=+3, falling back from L0 all
         the way to level 6/10 (Co, N_donor), losing donor-subclass and
         geometry specificity TOGETHER rather than just charge, because
         every intermediate fallback level still conditions on the exact
         (unclamped) charge value and none of them exist for charge=3).
         Affects any complex with |charge|>1 (also e.g. PtCl4/ZnCl4,
         charge=-2, though those happen not to show the symptom since
         their single homoleptic donor type means all sites already share
         one fallback target regardless of level). See
         _Q1_CHARGE_BUCKETS/_clamp_q1_charge() just below. Independent of,
         and unrelated to, the same-session Q2/Q3 GMM-cos refit and Q3
         mode-commit work (2.7.0/2.8.0) -- this is a Q1-only, charge-
         dimension-only fix.
2.8.0 : _score_components/score/score_breakdown gain an optional'''
do_replace("C0-version-history", OLD_TAIL, NEW_TAIL)

# ---------------------------------------------------------------------------
# C1: module-level charge-bucket constant + clamp helper, right after the
# existing _Q3_OVERRIDE_NEG constant (2.8.0).
# ---------------------------------------------------------------------------
do_replace(
    "C1-charge-bucket-constant-and-helper",
    "_Q3_OVERRIDE_NEG: float = -1.0e30",
    "_Q3_OVERRIDE_NEG: float = -1.0e30\n\n"
    "#: 2.9.0 -- the only charge values present in the live constraints.json's\n"
    "#: Q1 entries, for every metal (verified by direct inspection). Query-time\n"
    "#: charge is clamped to the nearest of these before building a Q1 key --\n"
    "#: see _clamp_q1_charge() and its call site in Scorer.__init__.\n"
    "_Q1_CHARGE_BUCKETS: Tuple[int, ...] = (-1, 0, 1)\n\n\n"
    "def _clamp_q1_charge(charge: int) -> int:\n"
    '    """\n'
    "    Clamp a complex's raw charge to the nearest bucket the live Q1\n"
    "    database actually contains (2.9.0 -- see module changelog). Without\n"
    "    this, any |charge|>1 complex (e.g. [Co(NH3)6]3+) fails to match at\n"
    "    every fallback level that still conditions on charge, jumping\n"
    "    straight to the fully generic (metal, donor_family) 2-tuple instead\n"
    '    of a donor-subclass/geometry-specific one at a nearby charge state.\n'
    '    """\n'
    "    return min(_Q1_CHARGE_BUCKETS, key=lambda b: abs(b - charge))",
)

# ---------------------------------------------------------------------------
# C2: query-time construction site -- clamp charge right after reading it.
# ---------------------------------------------------------------------------
do_replace(
    "C2-clamp-at-construction",
    "        spin = mol.spin_state\n"
    "        charge = mol.charge\n",
    "        spin = mol.spin_state\n"
    "        # 2.9.0: clamp to the DB's actual charge buckets -- see\n"
    "        # _clamp_q1_charge() docstring / module changelog.\n"
    "        charge = _clamp_q1_charge(int(mol.charge))\n",
)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
