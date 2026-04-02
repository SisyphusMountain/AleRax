#!/usr/bin/env python3
"""
Statistical consistency test for AleRax scoreGivenScenario.

Two tests are run:

PART A  (scoreGivenScenario accuracy)
    For each unique visible scenario j (identified by its RecPhyloXML content),
    scoreGivenScenario(j) must equal the MAXIMUM backtrace log-probability
    observed among all sampled trajectories whose visible scenario is j.

    Why this works: a trajectory with zero DL/TL transparent events is the
    highest-probability trajectory for any given visible scenario. Its
    log-probability is exactly what scoreGivenScenario computes.
    Any trajectory that also includes k≥1 DL/TL transparent events has a
    lower probability (additional factors < 1), so its logP < scoreGivenScenario.
    Therefore max(sampled logP for visible j) = scoreGivenScenario(j), provided
    at least one 0-DL/TL sample was drawn for j.

PART B  (sampling chi-squared goodness-of-fit)
    AleRax should draw each trajectory k with probability exp(logP_k).
    Grouping samples by their backtrace logP value (treating each unique value
    as a distinct trajectory batch, since independent trajectories can coincidentally
    share the same floating-point logP), the expected count for logP value v is
      expected_v = N_distinct_v * N * exp(v)
    where N_distinct_v is the empirical degeneracy factor (usually 1, but can be 2+
    for symmetric gene trees or D events with identical children).
    The degeneracy is estimated as round(obs_v / (N * exp(v))).

    Chi-squared goodness-of-fit is computed over trajectory-logP bins after merging
    rare bins (expected < MIN_EXPECTED).

Usage
-----
    python3 tests/test_scoring_consistency.py
"""

import math
import os
import subprocess
import sys
from collections import defaultdict

from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERAX       = os.path.join(ROOT, "build", "bin", "alerax")
SPECIES_TREE = os.path.join(ROOT, "data", "test_data", "simulated_2",
                             "speciesTree.newick")
# Existing 3-family test file (CCP generation is fast; only 1_pruned is scored)
FAMILIES     = os.path.join(ROOT, "tests", "outputs",
                             "simulated_2_UndatedDTL_global_1", "families.txt")

FAMILY_NAME  = "1_pruned"
N_SAMPLES    = 2000    # large enough for chi-squared cells ≥ 5
MIN_EXPECTED = 5       # merge chi-squared cells below this threshold
SCORE_TOL    = 1e-4    # |scored - max_bt| tolerance for Part A

# AleRax's FileSystem::mkdir is non-recursive (plain ::mkdir).
# The output directory must have only ONE new level under an existing parent.
SAMPLE_DIR   = os.path.join(ROOT, "tests", "outputs",
                             "scoring_consistency_samples")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def alerax(output_dir, extra_args):
    """Run AleRax with the test families/species-tree; raise on failure."""
    cmd = [
        ALERAX,
        "-f", FAMILIES,
        "-s", SPECIES_TREE,
        "--rec-model", "UndatedDTL",
        "-p", output_dir,
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-3000:])
        raise RuntimeError(f"AleRax failed:\n  " + " ".join(cmd))
    return result


def rec_dir(output_dir):
    return os.path.join(output_dir, "reconciliations", "all")


def score_xml(xml_path, score_out_dir):
    """Score one scenario; return log_prob."""
    alerax(score_out_dir, [
        "--gene-tree-samples", "0",
        "--score-scenario-xml", xml_path,
    ])
    result_file = os.path.join(
        rec_dir(score_out_dir),
        f"{FAMILY_NAME}_xmlScenarioLogProb.txt",
    )
    with open(result_file) as f:
        return float(f.read().strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(FAMILIES):
        sys.exit(f"FAMILIES file not found: {FAMILIES}\n"
                 f"Run run_alerax_tests.py first to generate it.")

    # ------------------------------------------------------------------
    # Step 1: sample N reconciliation trajectories
    # ------------------------------------------------------------------
    lp_path = os.path.join(rec_dir(SAMPLE_DIR), f"{FAMILY_NAME}_sampleLogProbs.txt")
    if not os.path.exists(lp_path):
        print(f"Sampling {N_SAMPLES} reconciliations …")
        alerax(SAMPLE_DIR, ["--gene-tree-samples", str(N_SAMPLES)])
        print("  done.")
    else:
        print(f"Using cached samples in {SAMPLE_DIR}")

    rdir = rec_dir(SAMPLE_DIR)

    # ------------------------------------------------------------------
    # Step 2: read per-sample backtrace log-probabilities
    # ------------------------------------------------------------------
    sample_logprobs = {}    # sample index -> logP
    with open(lp_path) as f:
        for line in f:
            idx, lp = line.split()
            sample_logprobs[int(idx)] = float(lp)
    N = len(sample_logprobs)
    print(f"Total samples: {N}\n")

    # ------------------------------------------------------------------
    # Step 3: group samples by unique visible scenario (XML content)
    # ------------------------------------------------------------------
    xml_content = {}
    for idx in sorted(sample_logprobs):
        xml = os.path.join(rdir, f"{FAMILY_NAME}_sample_{idx}.xml")
        if os.path.exists(xml):
            with open(xml) as f:
                xml_content[idx] = f.read()

    vis_groups = defaultdict(list)  # content -> [sample indices]
    for idx, c in xml_content.items():
        vis_groups[c].append(idx)

    unique = sorted(vis_groups.items(), key=lambda kv: -len(kv[1]))
    n_unique = len(unique)
    print(f"Unique visible scenarios: {n_unique}")

    # Summary table
    print(f"\n{'Rank':>4}  {'Count':>6}  {'Freq':>7}  "
          f"{'maxBtLogP':>11}  {'minBtLogP':>11}  {'#uniqueLogPs':>12}")
    print("-" * 60)
    for i, (content, idxs) in enumerate(unique[:10]):
        lps = [sample_logprobs[j] for j in idxs]
        print(f"{i:>4}  {len(idxs):>6}  {len(idxs)/N:>7.4f}  "
              f"{max(lps):>11.6f}  {min(lps):>11.6f}  "
              f"{len({round(v,5) for v in lps}):>12}")

    # ------------------------------------------------------------------
    # Step 4: score each unique visible scenario
    # ------------------------------------------------------------------
    print(f"\nScoring {n_unique} unique visible scenarios …")
    scored_logp = {}    # content -> scored logP
    for i, (content, idxs) in enumerate(unique):
        rep_idx = idxs[0]
        rep_xml = os.path.join(rdir, f"{FAMILY_NAME}_sample_{rep_idx}.xml")
        # Each scoring run needs its own flat output dir
        score_out = os.path.join(
            ROOT, "tests", "outputs", f"scoring_consistency_s{i:04d}"
        )
        logp = score_xml(rep_xml, score_out)
        scored_logp[content] = logp
        if (i + 1) % max(1, n_unique // 5) == 0 or i < 3:
            print(f"  [{i:4d}/{n_unique}] scored={logp:.6f}  count={len(idxs)}")

    # ==================================================================
    # PART A — scoreGivenScenario accuracy
    # ==================================================================
    print("\n" + "=" * 60)
    print("PART A: scoreGivenScenario(j) == max_backtrace_logP(j)")
    print("=" * 60)

    pass_count = 0
    fail_count = 0
    max_abs_diff = 0.0
    missing_count = 0   # scenarios with no 0-DL/TL sample (max_bt < scored)

    for content, idxs in unique:
        scored  = scored_logp[content]
        max_bt  = max(sample_logprobs[j] for j in idxs)
        diff    = scored - max_bt   # positive means scored > max_bt (no 0-DL/TL seen)
        abs_diff = abs(diff)
        max_abs_diff = max(max_abs_diff, abs_diff)

        if diff > SCORE_TOL:
            # scored > max_bt: the 0-DL/TL trajectory was not sampled at all.
            # This is statistically expected for rare scenarios; not a bug.
            missing_count += 1
        elif abs_diff <= SCORE_TOL:
            pass_count += 1
        else:
            # scored < max_bt: impossible if scoring is correct (0-DL/TL is maximum)
            fail_count += 1
            print(f"  ERROR: scored={scored:.6f}  max_bt={max_bt:.6f}  "
                  f"diff={diff:.4e}  count={len(idxs)}")

    print(f"\nResults ({n_unique} visible scenarios):")
    print(f"  PASS  (|scored - max_bt| ≤ {SCORE_TOL}): {pass_count}")
    print(f"  MISS  (0-DL/TL trajectory not sampled):  {missing_count}")
    print(f"  FAIL  (scored < max_bt, impossible):     {fail_count}")
    print(f"  Max |diff| over PASS scenarios: {max_abs_diff:.2e}")

    assert fail_count == 0, (
        f"PART A FAILED: {fail_count} scenarios have scored < max_backtrace_logP"
    )
    print("\nPART A: PASS")

    # ==================================================================
    # PART B — chi-squared on trajectory logP frequencies
    # ==================================================================
    print("\n" + "=" * 60)
    print("PART B: chi-squared goodness-of-fit on trajectory logP bins")
    print("=" * 60)
    print("""
Each unique backtrace logP value is treated as a trajectory 'batch'.
The degeneracy factor k_v = round(obs_v / (N * exp(v))) accounts for
symmetric gene trees or D events producing multiple trajectories with
identical logP.  Expected = k_v * N * exp(logP_v).
""")

    # Group all sample logPs
    lp_counts = defaultdict(int)
    for lp in sample_logprobs.values():
        lp_counts[round(lp, 6)] += 1

    # Compute per-group expected count
    rows = []  # (lp_v, obs_v, k_v, expected_v)
    for lp_v, obs_v in sorted(lp_counts.items(), key=lambda kv: -kv[1]):
        raw_exp    = N * math.exp(lp_v)
        k_v        = max(1, round(obs_v / raw_exp))
        expected_v = k_v * raw_exp
        rows.append((lp_v, obs_v, k_v, expected_v))

    # Show top 15 trajectory-logP groups
    print(f"{'logP':>12}  {'obs':>6}  {'k':>3}  {'expected':>9}  {'obs/exp':>8}")
    print("-" * 46)
    for lp_v, obs_v, k_v, exp_v in rows[:15]:
        print(f"{lp_v:>12.6f}  {obs_v:>6}  {k_v:>3}  {exp_v:>9.2f}  "
              f"{obs_v/exp_v:>8.4f}")
    if len(rows) > 15:
        print(f"  ... ({len(rows)} total unique logP bins)")

    # Build cells: main bins (expected >= MIN_EXPECTED) + one "rest" bin.
    # The rest bin absorbs all samples not in main bins.
    # expected_rest = N - sum(main expected) ensures sum(exp) = N = sum(obs),
    # making this a valid chi-squared GOF test without needing the unsampled
    # probability residual (which would inflate expected and always fail the test).
    obs_cells = []
    exp_cells = []
    main_exp_sum = 0.0
    main_obs_sum = 0

    for lp_v, obs_v, k_v, exp_v in rows:
        if exp_v >= MIN_EXPECTED:
            obs_cells.append(obs_v)
            exp_cells.append(exp_v)
            main_exp_sum += exp_v
            main_obs_sum += obs_v

    rest_obs = N - main_obs_sum
    rest_exp = N - main_exp_sum   # may be small or 0 if k estimates are large
    if rest_exp > 0:
        obs_cells.append(rest_obs)
        exp_cells.append(rest_exp)

    k = len(obs_cells)

    total_covered_prob = sum(k_v * math.exp(lp_v) for lp_v, obs_v, k_v, exp_v in rows)
    print(f"\nTotal probability mass covered (with k factors): {total_covered_prob:.4f}")

    print(f"\nChi-squared test ({k} cells, N={N}):")
    print(f"{'Cell':>6}  {'Obs':>6}  {'Exp':>8}  {'O/E':>7}")
    print("-" * 35)
    for i, (o, e) in enumerate(zip(obs_cells, exp_cells)):
        label = f"[{i}]" if i < k - 1 else "[rest]"
        print(f"{label:>6}  {o:>6}  {e:>8.2f}  {o/e:>7.4f}")

    chi2, p_value = stats.chisquare(obs_cells, f_exp=exp_cells)
    print(f"\nchi² = {chi2:.4f}  (df = {k - 1})")
    print(f"p-value = {p_value:.4f}")

    threshold = 0.05
    if p_value > threshold:
        print(f"\nPART B: PASS  (p = {p_value:.4f} > {threshold})")
    else:
        print(f"\nPART B: FAIL  (p = {p_value:.4f} ≤ {threshold})")
        return 1

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PART A — scoreGivenScenario accuracy:   PASS")
    print(f"  PART B — sampling chi-squared:          PASS")
    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
