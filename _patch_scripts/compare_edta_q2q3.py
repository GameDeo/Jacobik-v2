#!/usr/bin/env python3
"""
Quantify how much the Q2/Q3 GMM-cos refit changed the angular
distributions actually relevant to the EDTA test case, to check whether
it's a plausible root cause for the "carboxylates have uneven bond
lengths / N anchoring is uneven" regression the user reported.

For each relevant key, report:
  OLD (VMM): kappa, effective sigma_deg = degrees(1/sqrt(kappa)) (large-kappa
             approx), peak log-density.
  NEW (GMM_cos): var (cos-space), the mean converted to degrees, and the
             LOCAL effective sigma_deg at that mean via the Jacobian
             (sigma_cos / |sin(theta)|), plus peak log-density.

This lets us see, per component, whether a given angular preference got
tighter (smaller effective sigma_deg) or looser, and by how much.
"""
import glob
import json
import math

bak = sorted(glob.glob("src/jacobik/constraints.json.bak_pre_q2q3_refit_*"))[-1]
old = json.load(open(bak))
new = json.load(open("src/jacobik/constraints.json"))

EDTA_DONOR_CLASSES = {"N_amine", "O_carboxylate"}


def relevant_q2_keys():
    out = []
    for k in new:
        if not k.startswith("Q2:"):
            continue
        if "'Co'" not in k:
            continue
        if "octahedral" not in k:
            continue
        # crude parse of the tuple repr to check both donor subclasses
        # are in the EDTA donor set
        import ast

        tup = ast.literal_eval(k.split(":", 1)[1])
        _, dist_class, metal, sub_a, sub_b = tup
        if sub_a in EDTA_DONOR_CLASSES and sub_b in EDTA_DONOR_CLASSES:
            out.append((k, dist_class, sub_a, sub_b))
    return out


def relevant_q3_keys():
    out = []
    for k in new:
        if not k.startswith("Q3:"):
            continue
        if "'Co'" not in k or "octahedral" not in k:
            continue
        import ast

        tup = ast.literal_eval(k.split(":", 1)[1])
        sub, metal, geom = tup
        if sub in EDTA_DONOR_CLASSES:
            out.append((k, sub))
    return out


def old_vmm_summary(entry):
    parts = []
    for m, kappa, w in zip(entry["means"], entry["kappas"], entry["weights"]):
        sigma_deg = math.degrees(1.0 / math.sqrt(max(kappa, 1e-6)))
        log_norm = math.log(2 * math.pi) + math.log(max(_i0e(kappa), 1e-300)) + kappa
        peak_lp = kappa - log_norm  # at cos_diff=1
        parts.append(
            f"mu={m:6.1f}deg kappa={kappa:9.2f} sigma~={sigma_deg:5.1f}deg peak_lp={peak_lp:5.2f} w={w:.2f}"
        )
    return " | ".join(parts)


def _i0e(k):
    # crude but fine for reporting: use math via series/scipy if available
    try:
        from scipy.special import i0e

        return float(i0e(k))
    except Exception:
        return math.exp(-k) * math.e**0  # fallback, unused path


def new_gmm_summary(entry):
    parts = []
    for m, v, w in zip(entry["means"], entry["vars"], entry["weights"]):
        m_clamped = max(-1.0, min(1.0, m))
        theta_deg = math.degrees(math.acos(m_clamped))
        sin_theta = max(math.sin(math.radians(theta_deg)), 1e-6)
        sigma_cos = math.sqrt(max(v, 0.0025))
        sigma_deg_local = math.degrees(sigma_cos / sin_theta)
        peak_lp = -0.5 * math.log(2 * math.pi) - 0.5 * math.log(max(v, 0.0025))
        parts.append(
            f"mu={theta_deg:6.1f}deg var_cos={v:.5f} sigma_deg~={sigma_deg_local:6.1f}deg "
            f"peak_lp={peak_lp:5.2f} w={w:.2f}"
        )
    return " | ".join(parts)


print("=" * 100)
print("Q2 keys relevant to EDTA (Co, octahedral, N_amine/O_carboxylate pairs)")
print("=" * 100)
for k, dist_class, sub_a, sub_b in sorted(relevant_q2_keys()):
    print(f"\n{k}")
    if k in old:
        print(f"  OLD (n={old[k]['n']:5d} k={old[k]['k']}): {old_vmm_summary(old[k])}")
    else:
        print("  OLD: <missing>")
    if k in new:
        print(f"  NEW (n={new[k]['n']:5d} k={new[k]['k']}): {new_gmm_summary(new[k])}")
    else:
        print("  NEW: <missing>")

print()
print("=" * 100)
print("Q3 keys relevant to EDTA (Co, octahedral, N_amine/O_carboxylate)")
print("=" * 100)
for k, sub in sorted(relevant_q3_keys()):
    print(f"\n{k}")
    if k in old:
        print(f"  OLD (n={old[k]['n']:5d} k={old[k]['k']}): {old_vmm_summary(old[k])}")
    else:
        print("  OLD: <missing>")
    if k in new:
        print(f"  NEW (n={new[k]['n']:5d} k={new[k]['k']}): {new_gmm_summary(new[k])}")
    else:
        print("  NEW: <missing>")
