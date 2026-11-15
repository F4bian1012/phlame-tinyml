#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks the statistics behind the tables of the paper (src/build_ladder_tables.py):
macro-F1, the exact McNemar test, the Holm correction, the paired percentile
bootstrap and the t interval of the repeatability runs.

The expected values do not come from the functions under test. They are
either worked out by hand in the comments, or computed here in exact
fractions from the textbook definitions:

  macro-F1    mean over the two classes of 2PR / (P + R), from precision and
              recall; the script uses 2TP / (2TP + FP + FN)
  McNemar     two-sided exact p = min(1, 2 * sum_{i <= min(b, c)} C(b + c, i) / 2^(b + c))
              on the b + c discordant images
  Holm        the k-th smallest of m p-values is multiplied by m - k + 1, and
              each adjusted value is at least the one before it and at most 1
  bootstrap   for every row of resample indices, the difference of the two
              macro-F1 values on that same resample; the interval is the 2.5th
              and 97.5th percentile with linear interpolation

The last class checks the published tables themselves against those
definitions and against the per-image result files: every McNemar p-value
from its two discordant counts, every Holm column from its raw column, the
Fisher p-values of the per-class table, and the macro-F1, the errors and the
discordant counts of the first table recounted from the prediction files.

Needs numpy and scipy (not TensorFlow); skipped when either is missing. The
last class needs neither.

    python -m unittest discover -s tests -v
"""
import csv
import importlib.util
import random
import sys
import unittest
from fractions import Fraction
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

READY = all(importlib.util.find_spec(name) for name in ("numpy", "scipy"))
if READY:
    import numpy as np

    import build_ladder_tables as tables


# --------------------------------------------------------------------------- #
# The definitions, in exact arithmetic
# --------------------------------------------------------------------------- #
def f1_macro_reference(y, p):
    """Macro-F1 of two classes from precision and recall. A class with no true positive scores 0."""
    total = Fraction(0)
    for c in (0, 1):
        true_positive = sum(1 for a, b in zip(y, p) if a == c and b == c)
        if true_positive == 0:
            continue
        precision = Fraction(true_positive, sum(1 for b in p if b == c))
        recall = Fraction(true_positive, sum(1 for a in y if a == c))
        total += 2 * precision * recall / (precision + recall)
    return total / 2


def mcnemar_reference(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return Fraction(1)
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2 ** n))


def holm_reference(ps):
    """Holm step-down adjusted p-values, in the order given."""
    m = len(ps)
    adjusted, running = [None] * m, 0
    for rank, i in enumerate(sorted(range(m), key=lambda k: ps[k])):
        running = max(running, (m - rank) * ps[i])
        adjusted[i] = min(1, running)
    return adjusted


def fisher_reference(e1, n1, e2, n2):
    """Two-sided Fisher exact p of the 2x2 table [[e1, n1 - e1], [e2, n2 - e2]]: the sum of the
    probabilities of the tables with the same margins that are no more likely than the observed one."""
    errors, total = e1 + e2, n1 + n2

    def weight(k):   # number of ways to place k of the errors in the first row
        return comb(n1, k) * comb(n2, errors - k)

    low, high = max(0, errors - n2), min(n1, errors)
    observed = weight(e1)
    tail = sum(weight(k) for k in range(low, high + 1) if weight(k) <= observed)   # integers: ties are exact
    return Fraction(tail, comb(total, errors))


def percentile_reference(values, q):
    """Percentile with linear interpolation between the two nearest order statistics."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    below = int(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (position - below) * (ordered[above] - ordered[below])


def outcomes(both_right, only_a, only_b, both_wrong, seed=0):
    """Labels and the predictions of two conditions with the given numbers of images where both
    are right, only the first, only the second, or neither; the images are in random order."""
    rng = random.Random(seed)
    kinds = ["rr"] * both_right + ["rw"] * only_a + ["wr"] * only_b + ["ww"] * both_wrong
    rng.shuffle(kinds)
    y = [rng.randint(0, 1) for _ in kinds]
    pa = [label if kind[0] == "r" else 1 - label for label, kind in zip(y, kinds)]
    pb = [label if kind[1] == "r" else 1 - label for label, kind in zip(y, kinds)]
    return y, pa, pb


def arrays(*lists):
    return [np.array(values, dtype=np.int64) for values in lists]


# --------------------------------------------------------------------------- #
# The functions of the script
# --------------------------------------------------------------------------- #
@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class MacroF1Test(unittest.TestCase):
    def test_worked_example(self):
        y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        p = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
        # class 0: TP 3, FP 2, FN 1 -> F1 = 6 / 9 = 2/3
        # class 1: TP 4, FP 1, FN 2 -> F1 = 8 / 11
        # macro  : (2/3 + 8/11) / 2 = 23/33
        self.assertAlmostEqual(float(tables.f1_macro(*arrays(y, p))), 23 / 33, places=15)
        self.assertEqual(f1_macro_reference(y, p), Fraction(23, 33))

    def test_perfect_and_inverted_predictions(self):
        y = [0, 1, 1, 0, 1, 0, 0, 1]
        self.assertEqual(float(tables.f1_macro(*arrays(y, y))), 1.0)
        self.assertEqual(float(tables.f1_macro(*arrays(y, [1 - v for v in y]))), 0.0)

    def test_a_constant_answer_on_balanced_data_scores_one_third(self):
        # class 0: TP 5, FP 5, FN 0 -> 10 / 15 = 2/3; class 1: nothing right -> 0; macro 1/3.
        # Accuracy is 0.5 for the same predictions, which is why the tables report macro-F1.
        y = [0] * 5 + [1] * 5
        self.assertAlmostEqual(float(tables.f1_macro(*arrays(y, [0] * 10))), 1 / 3, places=15)

    def test_a_class_that_never_appears_counts_as_zero(self):
        # Only class 0 is in the labels and in the predictions: its F1 is 1 and the other
        # class has no true positive, false positive or false negative. It scores 0, not 1.
        self.assertEqual(float(tables.f1_macro(*arrays([0, 0, 0], [0, 0, 0]))), 0.5)

    def test_agrees_with_precision_and_recall_on_random_cases(self):
        rng = random.Random(20261115)
        for _ in range(200):
            n = rng.randint(1, 40)
            y = [rng.randint(0, 1) for _ in range(n)]
            p = [rng.randint(0, 1) for _ in range(n)]
            self.assertAlmostEqual(float(tables.f1_macro(*arrays(y, p))), float(f1_macro_reference(y, p)),
                                   places=14, msg="y=%s p=%s" % (y, p))

    def test_is_computed_row_by_row_on_a_matrix(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, (7, 25))
        p = rng.integers(0, 2, (7, 25))
        together = tables.f1_macro(y, p)
        self.assertEqual(together.shape, (7,))
        for i in range(7):
            self.assertEqual(float(together[i]), float(tables.f1_macro(y[i], p[i])))
            self.assertAlmostEqual(float(together[i]), float(f1_macro_reference(y[i].tolist(), p[i].tolist())),
                                   places=14)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "the cross-check needs scikit-learn")
    def test_agrees_with_scikit_learn(self):
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(11)
        for _ in range(50):
            y, p = rng.integers(0, 2, 60), rng.integers(0, 2, 60)
            expected = f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)
            self.assertAlmostEqual(float(tables.f1_macro(y, p)), float(expected), places=14)


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class McNemarTest(unittest.TestCase):
    """paired(): discordant counts, exact p, difference of macro-F1 and agreement."""

    def paired(self, both_right, only_a, only_b, both_wrong, seed=0):
        y, pa, pb = outcomes(both_right, only_a, only_b, both_wrong, seed)
        identity = np.arange(len(y))[np.newaxis, :]      # one "resample": the images themselves
        return y, pa, pb, tables.paired(*arrays(y, pa, pb), identity)

    def test_p_values_worked_by_hand(self):
        # p = min(1, 2 * P(X <= min(b, c))) with X ~ Binomial(b + c, 1/2)
        cases = [
            ((1, 5), 14 / 64),          # 2 * (1 + 6) / 2^6
            ((0, 5), 2 / 32),           # 2 * 1 / 2^5
            ((5, 0), 2 / 32),
            ((2, 8), 112 / 1024),       # 2 * (1 + 10 + 45) / 2^10
            ((3, 3), 1.0),              # 2 * (1 + 6 + 15 + 20) / 2^6 = 84/64, capped at 1
            ((0, 1), 1.0),              # 2 * 1 / 2
            ((0, 0), 1.0),              # no discordant image: nothing to test
            ((17, 4), 15094 / 2097152),  # 2 * (1 + 21 + 210 + 1330 + 5985) / 2^21
        ]
        for (b, c), expected in cases:
            _, _, _, result = self.paired(20, b, c, 6)
            self.assertEqual((result["only_a"], result["only_b"]), (b, c))
            self.assertAlmostEqual(result["p"], expected, places=14, msg="b=%d c=%d" % (b, c))
            self.assertAlmostEqual(float(mcnemar_reference(b, c)), expected, places=15)

    def test_p_value_agrees_with_the_closed_form(self):
        rng = random.Random(7)
        for _ in range(60):
            b, c = rng.randint(0, 40), rng.randint(0, 40)
            _, _, _, result = self.paired(rng.randint(0, 30), b, c, rng.randint(0, 30), seed=rng.random())
            self.assertAlmostEqual(result["p"], float(mcnemar_reference(b, c)), places=13, msg="b=%d c=%d" % (b, c))

    def test_only_the_discordant_images_count(self):
        few = self.paired(3, 2, 9, 1)[3]
        many = self.paired(300, 2, 9, 150)[3]
        self.assertEqual(few["p"], many["p"])

    def test_swapping_the_conditions_keeps_p_and_mirrors_the_rest(self):
        y, pa, pb = outcomes(30, 4, 11, 5, seed=2)
        identity = np.arange(len(y))[np.newaxis, :]
        forward = tables.paired(*arrays(y, pa, pb), identity)
        backward = tables.paired(*arrays(y, pb, pa), identity)
        self.assertEqual(forward["p"], backward["p"])
        self.assertEqual((forward["only_a"], forward["only_b"]), (backward["only_b"], backward["only_a"]))
        self.assertAlmostEqual(forward["d_f1"], -backward["d_f1"], places=15)

    def test_difference_and_agreement(self):
        y, pa, pb, result = self.paired(30, 4, 11, 5, seed=3)
        expected = f1_macro_reference(y, pb) - f1_macro_reference(y, pa)
        self.assertAlmostEqual(result["d_f1"], float(expected), places=14)
        # two binary predictions are equal when both are right or both are wrong: 35 of 50 images
        self.assertAlmostEqual(result["agreement_pct"], 70.0, places=12)


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class HolmTest(unittest.TestCase):
    def holm(self, ps):
        return [float(v) for v in tables.holm(ps)]

    def assert_close(self, got, expected):
        self.assertEqual(len(got), len(expected))
        for a, b in zip(got, expected):
            self.assertAlmostEqual(a, b, places=14)

    def test_worked_examples(self):
        # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> never decreasing: 0.06
        self.assert_close(self.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
        # sorted 0.01, 0.03, 0.04 -> 0.03, 0.06, 0.04 -> 0.03, 0.06, 0.06
        self.assert_close(self.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        # 0.2 x 3 = 0.6; 0.5 x 2 = 1.0; 0.9 x 1 = 0.9 -> 1.0 (not below the one before)
        self.assert_close(self.holm([0.2, 0.5, 0.9]), [0.6, 1.0, 1.0])
        # the product would be 2.4: an adjusted p-value is at most 1
        self.assert_close(self.holm([0.8, 0.9, 0.95]), [1.0, 1.0, 1.0])

    def test_one_value_is_left_as_it_is_and_ties_get_the_same(self):
        self.assert_close(self.holm([0.037]), [0.037])
        self.assert_close(self.holm([0.02, 0.02]), [0.04, 0.04])
        self.assertEqual(self.holm([]), [])

    def test_agrees_with_the_definition_on_random_families(self):
        rng = random.Random(5)
        for _ in range(100):
            ps = [Fraction(rng.randint(0, 1000), 1000) for _ in range(rng.randint(1, 12))]
            self.assert_close(self.holm([float(p) for p in ps]), [float(v) for v in holm_reference(ps)])

    def test_properties(self):
        rng = random.Random(9)
        for _ in range(50):
            ps = [rng.random() for _ in range(rng.randint(2, 10))]
            adjusted = self.holm(ps)
            m = len(ps)
            for raw, adj in zip(ps, adjusted):
                self.assertGreaterEqual(adj, raw)                    # a correction never lowers a p-value
                self.assertLessEqual(adj, min(1.0, m * raw) + 1e-15)  # and is never above Bonferroni
            by_raw = sorted(zip(ps, adjusted))
            self.assertEqual([a for _, a in by_raw], sorted(a for _, a in by_raw))   # same order as the raw values
            # the result follows its p-value when the family is given in another order
            order = list(range(m))
            rng.shuffle(order)
            shuffled = self.holm([ps[i] for i in order])
            self.assert_close(shuffled, [adjusted[i] for i in order])


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class PairedBootstrapTest(unittest.TestCase):
    """The interval of paired(): both conditions are scored on the same resample of the images."""

    def setUp(self):
        self.y, self.pa, self.pb = outcomes(40, 6, 13, 9, seed=4)
        self.n = len(self.y)

    def resamples(self, seed, b=200):
        # as main() draws them: one row of image indices per resample, with replacement
        return np.random.default_rng(seed).integers(0, self.n, (b, self.n), dtype=np.int32)

    def expected_interval(self, idx):
        deltas = []
        for row in idx.tolist():
            y, pa, pb = ([values[i] for i in row] for values in (self.y, self.pa, self.pb))
            deltas.append(float(f1_macro_reference(y, pb) - f1_macro_reference(y, pa)))
        return percentile_reference(deltas, 2.5), percentile_reference(deltas, 97.5)

    def test_interval_is_the_percentiles_of_the_paired_differences(self):
        idx = self.resamples(1)
        result = tables.paired(*arrays(self.y, self.pa, self.pb), idx)
        low, high = self.expected_interval(idx)
        self.assertAlmostEqual(result["ci_lo"], low, places=12)
        self.assertAlmostEqual(result["ci_hi"], high, places=12)
        self.assertLess(result["ci_lo"], result["d_f1"])
        self.assertGreater(result["ci_hi"], result["d_f1"])

    def test_same_seed_gives_the_same_resamples_and_the_same_interval(self):
        first, again, other = self.resamples(1), self.resamples(1), self.resamples(2)
        self.assertTrue(np.array_equal(first, again))
        self.assertFalse(np.array_equal(first, other))
        a = tables.paired(*arrays(self.y, self.pa, self.pb), first)
        b = tables.paired(*arrays(self.y, self.pa, self.pb), again)
        c = tables.paired(*arrays(self.y, self.pa, self.pb), other)
        self.assertEqual(a, b)
        self.assertNotEqual((a["ci_lo"], a["ci_hi"]), (c["ci_lo"], c["ci_hi"]))
        # what does not come from the resamples does not depend on the seed
        for key in ("d_f1", "only_a", "only_b", "p", "agreement_pct"):
            self.assertEqual(a[key], c[key])

    def test_a_condition_against_itself_has_no_spread(self):
        # If the two conditions were scored on different resamples the interval would have
        # a width; on the same resample every difference is exactly zero.
        result = tables.paired(*arrays(self.y, self.pa, self.pa), self.resamples(1))
        self.assertEqual((result["d_f1"], result["ci_lo"], result["ci_hi"]), (0.0, 0.0, 0.0))
        self.assertEqual((result["only_a"], result["only_b"], result["p"]), (0, 0, 1.0))

    def test_the_paired_interval_is_narrower_than_two_independent_resamples_would_give(self):
        # The two conditions agree on 49 of 68 images; scoring them on the same resample removes
        # the part of the variation they share.
        idx_a, idx_b = self.resamples(1, b=2000), self.resamples(2, b=2000)
        y, pa, pb = arrays(self.y, self.pa, self.pb)
        paired = tables.paired(y, pa, pb, idx_a)
        unpaired = tables.f1_macro(y[idx_b], pb[idx_b]) - tables.f1_macro(y[idx_a], pa[idx_a])
        width_unpaired = float(np.percentile(unpaired, 97.5) - np.percentile(unpaired, 2.5))
        self.assertLess(paired["ci_hi"] - paired["ci_lo"], width_unpaired)

    def test_resampling_the_images_as_they_are_gives_the_point_estimate(self):
        identity = np.tile(np.arange(self.n), (5, 1))
        result = tables.paired(*arrays(self.y, self.pa, self.pb), identity)
        self.assertEqual((result["ci_lo"], result["ci_hi"]), (result["d_f1"], result["d_f1"]))


@unittest.skipUnless(READY, "build_ladder_tables.py needs numpy and scipy")
class TIntervalTest(unittest.TestCase):
    def test_worked_examples(self):
        # 1, 2, 3, 4: mean 2.5; squared deviations 2.25 + 0.25 + 0.25 + 2.25 = 5, over n - 1 = 3;
        # sd = sqrt(5/3); t(0.975, 3 d.f.) = 3.182446 (t table); half-width = t * sd / sqrt(4)
        mean, sd, half = tables.t_halfwidth([1, 2, 3, 4])
        self.assertAlmostEqual(mean, 2.5, places=15)
        self.assertAlmostEqual(sd, (5 / 3) ** 0.5, places=14)
        self.assertAlmostEqual(half, 3.182446305284263 * (5 / 3) ** 0.5 / 2, places=9)
        # 0 and 1: mean 0.5, sd = sqrt(0.5); t(0.975, 1 d.f.) = 12.706205
        mean, sd, half = tables.t_halfwidth([0, 1])
        self.assertAlmostEqual(mean, 0.5, places=15)
        self.assertAlmostEqual(sd, 0.5 ** 0.5, places=14)
        self.assertAlmostEqual(half, 12.706204736174694 * 0.5 ** 0.5 / 2 ** 0.5, places=8)


# --------------------------------------------------------------------------- #
# The published tables against the definitions and the per-image files
# --------------------------------------------------------------------------- #
TABLES_DIR = REPO / "results" / "tables"
RUN_SUFFIX = "+32+20+0.0001+160+120"       # the runs of the vehicle benchmark behind the tables


def read_table(name):
    with open(TABLES_DIR / name, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_predictions(level, arch):
    """{image: (true label, predicted label)} of a published run, or None when it does not exist."""
    name = "predictions_%s%s%s.csv" % (arch, RUN_SUFFIX, "" if level == "mil" else "_int8")
    path = REPO / "results" / level / name
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["Image"]: (int(row["TrueLabel"]), int(row["PredLabel"])) for row in csv.DictReader(fh, delimiter=";")}


class PublishedStatisticsTest(unittest.TestCase):
    """Standard library only: the numbers in results/tables/ against the definitions above."""

    def assert_holm(self, rows, raw, adjusted, table):
        with_value = [row for row in rows if row[raw] != ""]
        self.assertGreater(len(with_value), 0, "%s has no value in %s" % (table, raw))
        expected = holm_reference([float(row[raw]) for row in with_value])
        for row, value in zip(with_value, expected):
            self.assertAlmostEqual(float(row[adjusted]), float(value), places=12,
                                   msg="%s, %s of %s" % (table, adjusted, list(row.values())[0]))

    def test_every_mcnemar_p_value_follows_from_its_discordant_counts(self):
        checked = 0
        for table in ("table1_ladder.csv", "table2_decomposition.csv"):
            rows = read_table(table)
            columns = list(rows[0])
            for p_column in [c for c in columns if c.endswith("_p")]:
                prefix = p_column[:-len("_p")]
                counts = [c for c in columns if c.startswith(prefix + "_only_")]
                if len(counts) != 2:
                    continue      # not a McNemar test (the interaction of the second table is a sign test)
                for row in rows:
                    if row[p_column] == "":
                        continue  # an architecture that does not reach that level
                    b, c = int(row[counts[0]]), int(row[counts[1]])
                    self.assertAlmostEqual(float(row[p_column]), float(mcnemar_reference(b, c)), places=13,
                                           msg="%s, %s of %s (b=%d, c=%d)" % (table, p_column, row["arch"], b, c))
                    checked += 1
        self.assertGreaterEqual(checked, 40)

    def test_every_holm_column_follows_from_its_raw_column(self):
        families = 0
        for table in ("table1_ladder.csv", "table2_decomposition.csv"):
            rows = read_table(table)
            for adjusted in [c for c in rows[0] if c.endswith("_p_holm")]:
                self.assert_holm(rows, adjusted[:-len("_holm")], adjusted, table)
                families += 1
        self.assert_holm(read_table("table5_per_class.csv"), "p_fisher", "p_holm", "table5_per_class.csv")
        self.assert_holm(read_table("table6_gap_contrasts.csv"), "p_bootstrap", "p_holm", "table6_gap_contrasts.csv")
        self.assertGreaterEqual(families, 10)

    def test_fisher_p_values_follow_from_the_errors_of_each_class(self):
        for row in read_table("table5_per_class.csv"):
            n, e1, e2 = int(row["n"]), int(row["errors_0"]), int(row["errors_1"])
            # the test partition and its rig subset have the same number of images of each class
            n1 = n2 = n // 2
            self.assertEqual(n % 2, 0)
            self.assertAlmostEqual(float(row["recall_0"]), 1 - e1 / n1, places=14)
            self.assertAlmostEqual(float(row["recall_1"]), 1 - e2 / n2, places=14)
            self.assertAlmostEqual(float(row["p_fisher"]), float(fisher_reference(e1, n1, e2, n2)), places=12,
                                   msg="%s %s" % (row["arch"], row["level"]))

    def test_first_table_is_what_the_prediction_files_give(self):
        compared = 0
        for row in read_table("table1_ladder.csv"):
            arch = row["arch"]
            runs = {level: read_predictions(level, arch) for level in ("mil", "sil", "pil")}
            images = sorted(runs["mil"])
            self.assertEqual(len(images), 940)
            y = [runs["mil"][image][0] for image in images]
            self.assertEqual((y.count(0), y.count(1)), (470, 470))
            scores = {}
            for level, column in (("mil", "MIL"), ("sil", "SIL"), ("pil", "PIL")):
                if runs[level] is None:
                    self.assertEqual(row[column + "_f1"], "", "%s has a %s figure and no result file" % (arch, column))
                    continue
                self.assertEqual(sorted(runs[level]), images)
                self.assertEqual([runs[level][image][0] for image in images], y)
                predicted = [runs[level][image][1] for image in images]
                scores[level] = (f1_macro_reference(y, predicted), predicted)
                self.assertAlmostEqual(float(row[column + "_f1"]), float(scores[level][0]), places=13, msg=arch)
                self.assertEqual(int(row[column + "_errors"]), sum(1 for a, b in zip(y, predicted) if a != b), arch)
                compared += 1
            for first, second, prefix in (("mil", "sil", "quant"), ("sil", "pil", "exec")):
                if first not in scores or second not in scores:
                    continue
                pa, pb = scores[first][1], scores[second][1]
                only_first = sum(1 for t, a, b in zip(y, pa, pb) if a == t and b != t)
                only_second = sum(1 for t, a, b in zip(y, pa, pb) if a != t and b == t)
                self.assertEqual((int(row["%s_only_%s" % (prefix, first.upper())]),
                                  int(row["%s_only_%s" % (prefix, second.upper())])), (only_first, only_second), arch)
                self.assertAlmostEqual(float(row[prefix + "_d_f1"]), float(scores[second][0] - scores[first][0]),
                                       places=13, msg=arch)
        self.assertEqual(compared, 7 + 7 + 5)   # MIL and SIL of seven architectures, PIL of the five on the board


if __name__ == "__main__":
    unittest.main()
