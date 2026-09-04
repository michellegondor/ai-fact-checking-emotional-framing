from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = Path(
    "data/FEVER/final_batch/final_results.csv"
)

OUTPUT_DIR = Path(
    "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FCR_TABLE_FILE = OUTPUT_DIR / "fcr_by_emotion.csv"
FCR_LABEL_FILE = OUTPUT_DIR / "fcr_by_label.csv"
ACCURACY_FILE = OUTPUT_DIR / "accuracy_summary.csv"


EMOTIONS = [
    "positive",
    "negative",
    "fear",
    "anger",
]

EVIDENCE_CONDITIONS = [
    "present",
    "absent",
]


# ============================================================
# 1. LOAD FINAL DATA
# ============================================================

print("=" * 70)
print("FINAL EXPERIMENT ANALYSIS")
print("=" * 70)


df = pd.read_csv(
    RESULTS_FILE
)


print("\nRows:")
print(len(df))

print("\nUnique claims:")
print(df["id"].nunique())


if len(df) != 15000:
    raise ValueError(
        "Expected exactly 15000 experimental rows."
    )


if df["id"].nunique() != 1500:
    raise ValueError(
        "Expected exactly 1500 unique claims."
    )


# ============================================================
# 2. CHECK MISSING PREDICTIONS
# ============================================================

missing_predictions = (
    df["prediction"]
    .isna()
    .sum()
)


print("\nMissing predictions:")
print(missing_predictions)


if missing_predictions != 1:
    raise ValueError(
        "Expected exactly one known technical missing prediction."
    )


# ============================================================
# 3. CREATE NEUTRAL LOOKUP
# ============================================================

neutral = df[
    df["emotion"] == "neutral"
][
    [
        "id",
        "evidence_condition",
        "prediction",
    ]
].copy()


neutral = neutral.rename(
    columns={
        "prediction":
            "neutral_prediction"
    }
)


print("\nNeutral rows:")
print(len(neutral))


if len(neutral) != 3000:
    raise ValueError(
        "Expected 3000 neutral observations."
    )


if neutral[
    [
        "id",
        "evidence_condition",
    ]
].duplicated().any():

    raise ValueError(
        "Duplicate neutral comparison keys."
    )


# ============================================================
# 4. CREATE EMOTIONAL COMPARISON DATASET
# ============================================================

emotional = df[
    df["emotion"].isin(EMOTIONS)
].copy()


comparisons = emotional.merge(
    neutral,
    on=[
        "id",
        "evidence_condition",
    ],
    how="left",
    validate="many_to_one",
)


if comparisons[
    "neutral_prediction"
].isna().any():

    raise ValueError(
        "At least one emotional observation "
        "has no neutral baseline."
    )


# ============================================================
# 5. DEFINE VALID PAIRS AND FRAMING CHANGE
# ============================================================

comparisons[
    "valid_pair"
] = (
    comparisons["prediction"].notna()
    &
    comparisons[
        "neutral_prediction"
    ].notna()
)


comparisons[
    "framing_change"
] = pd.NA


comparisons.loc[
    comparisons["valid_pair"],
    "framing_change"
] = (
    comparisons.loc[
        comparisons["valid_pair"],
        "prediction"
    ]
    !=
    comparisons.loc[
        comparisons["valid_pair"],
        "neutral_prediction"
    ]
)


comparisons[
    "framing_change"
] = comparisons[
    "framing_change"
].astype("boolean")


# ============================================================
# 6. FCR BY EMOTION × EVIDENCE
# ============================================================

fcr_rows = []


for evidence_condition in EVIDENCE_CONDITIONS:

    for emotion in EMOTIONS:

        subset = comparisons[
            (
                comparisons[
                    "evidence_condition"
                ]
                == evidence_condition
            )
            &
            (
                comparisons[
                    "emotion"
                ]
                == emotion
            )
        ].copy()


        valid = subset[
            subset["valid_pair"]
        ]


        n_valid = len(valid)

        changes = int(
            valid[
                "framing_change"
            ].sum()
        )

        fcr = (
            changes / n_valid
            if n_valid > 0
            else float("nan")
        )


        fcr_rows.append(
            {
                "evidence_condition":
                    evidence_condition,

                "emotion":
                    emotion,

                "changes":
                    changes,

                "n_valid":
                    n_valid,

                "fcr":
                    fcr,

                "fcr_percent":
                    fcr * 100,
            }
        )


fcr_table = pd.DataFrame(
    fcr_rows
)


print("\n" + "=" * 70)
print("FRAMING CHANGE RATE BY EMOTION")
print("=" * 70)


for evidence_condition in EVIDENCE_CONDITIONS:

    print(
        f"\nEVIDENCE {evidence_condition.upper()}"
    )

    temp = fcr_table[
        fcr_table[
            "evidence_condition"
        ]
        == evidence_condition
    ]

    for _, row in temp.iterrows():

        print(
            f"{row['emotion']:8s} "
            f"{int(row['changes']):4d} / "
            f"{int(row['n_valid']):4d}   "
            f"{row['fcr_percent']:.3f}%"
        )


# ============================================================
# 7. AGGREGATED FCR PER CLAIM
# ============================================================

# Important:
# We calculate one sensitivity score per claim
# and evidence condition:
#
# number of emotional framings that changed
# divided by number of valid emotional comparisons.
#
# This preserves the claim as the unit of analysis.

claim_fcr = (
    comparisons[
        comparisons["valid_pair"]
    ]
    .groupby(
        [
            "id",
            "label",
            "evidence_condition",
        ]
    )["framing_change"]
    .agg(
        [
            "sum",
            "count",
        ]
    )
    .reset_index()
)


claim_fcr[
    "claim_fcr"
] = (
    claim_fcr["sum"]
    /
    claim_fcr["count"]
)


aggregate_fcr = (
    claim_fcr
    .groupby(
        "evidence_condition"
    )["claim_fcr"]
    .agg(
        [
            "mean",
            "count",
        ]
    )
)


print("\n" + "=" * 70)
print("AGGREGATED CLAIM-LEVEL FCR")
print("=" * 70)

print(
    aggregate_fcr
)


present_mean = (
    claim_fcr[
        claim_fcr[
            "evidence_condition"
        ]
        == "present"
    ]["claim_fcr"]
    .mean()
)


absent_mean = (
    claim_fcr[
        claim_fcr[
            "evidence_condition"
        ]
        == "absent"
    ]["claim_fcr"]
    .mean()
)


difference = (
    absent_mean
    -
    present_mean
)


print(
    "\nPrimary descriptive difference:"
)

print(
    "FCR absent - FCR present = "
    f"{difference:.6f}"
)

print(
    "Percentage-point difference = "
    f"{difference * 100:.3f} pp"
)


# ============================================================
# 8. FCR BY GROUND-TRUTH LABEL
# ============================================================

label_fcr = (
    claim_fcr
    .groupby(
        [
            "label",
            "evidence_condition",
        ]
    )["claim_fcr"]
    .agg(
        [
            "mean",
            "count",
        ]
    )
    .reset_index()
)


label_fcr[
    "fcr_percent"
] = (
    label_fcr["mean"]
    * 100
)


print("\n" + "=" * 70)
print("FCR BY GROUND-TRUTH LABEL")
print("=" * 70)

print(
    label_fcr.to_string(
        index=False
    )
)


# ============================================================
# 9. ACCURACY BY EVIDENCE × EMOTION
# ============================================================

accuracy_rows = []


for evidence_condition in EVIDENCE_CONDITIONS:

    for emotion in [
        "neutral",
        "positive",
        "negative",
        "fear",
        "anger",
    ]:

        subset = df[
            (
                df[
                    "evidence_condition"
                ]
                == evidence_condition
            )
            &
            (
                df[
                    "emotion"
                ]
                == emotion
            )
        ].copy()


        subset = subset[
            subset[
                "prediction"
            ].notna()
        ]


        n_valid = len(
            subset
        )

        correct_count = int(
            (
                subset[
                    "prediction"
                ]
                ==
                subset[
                    "label"
                ]
            ).sum()
        )

        accuracy = (
            correct_count
            / n_valid
        )


        accuracy_rows.append(
            {
                "evidence_condition":
                    evidence_condition,

                "emotion":
                    emotion,

                "correct":
                    correct_count,

                "n_valid":
                    n_valid,

                "accuracy":
                    accuracy,

                "accuracy_percent":
                    accuracy * 100,
            }
        )


accuracy_table = pd.DataFrame(
    accuracy_rows
)


print("\n" + "=" * 70)
print("ACCURACY BY CONDITION")
print("=" * 70)


for evidence_condition in EVIDENCE_CONDITIONS:

    print(
        f"\nEVIDENCE {evidence_condition.upper()}"
    )

    temp = accuracy_table[
        accuracy_table[
            "evidence_condition"
        ]
        == evidence_condition
    ]

    for _, row in temp.iterrows():

        print(
            f"{row['emotion']:8s} "
            f"{int(row['correct']):4d} / "
            f"{int(row['n_valid']):4d}   "
            f"{row['accuracy_percent']:.3f}%"
        )


# ============================================================
# 10. SAVE TABLES
# ============================================================

fcr_table.to_csv(
    FCR_TABLE_FILE,
    index=False
)


label_fcr.to_csv(
    FCR_LABEL_FILE,
    index=False
)


accuracy_table.to_csv(
    ACCURACY_FILE,
    index=False
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved:")

print(
    FCR_TABLE_FILE
)

print(
    FCR_LABEL_FILE
)

print(
    ACCURACY_FILE
)

print("=" * 70)
# ============================================================
# 11. PRIMARY H2 INFERENCE
# Complete-case paired analysis
# ============================================================

print("\n" + "=" * 70)
print("PRIMARY H2: DOES EVIDENCE REDUCE FRAMING SENSITIVITY?")
print("=" * 70)


# ------------------------------------------------------------
# Keep only claims with all 4 emotional comparisons
# in BOTH evidence conditions.
# ------------------------------------------------------------

complete_counts = (
    comparisons[
        comparisons["valid_pair"]
    ]
    .groupby(
        [
            "id",
            "evidence_condition",
        ]
    )
    .size()
    .unstack()
)


complete_ids = complete_counts[
    (complete_counts["present"] == 4)
    &
    (complete_counts["absent"] == 4)
].index


print("\nComplete claims used:")
print(len(complete_ids))


if len(complete_ids) != 1499:
    raise ValueError(
        "Expected exactly 1499 complete claims."
    )


# ------------------------------------------------------------
# Calculate one FCR score per claim and evidence condition
# ------------------------------------------------------------

complete_comparisons = comparisons[
    comparisons["id"].isin(
        complete_ids
    )
].copy()


claim_scores = (
    complete_comparisons
    .groupby(
        [
            "id",
            "evidence_condition",
        ]
    )["framing_change"]
    .mean()
    .unstack()
)


claim_scores[
    "difference"
] = (
    claim_scores["absent"]
    -
    claim_scores["present"]
)


observed_difference = (
    claim_scores["difference"].mean()
)


print("\nMean FCR with evidence:")
print(
    f"{claim_scores['present'].mean():.6f}"
)


print("\nMean FCR without evidence:")
print(
    f"{claim_scores['absent'].mean():.6f}"
)


print("\nObserved paired difference:")
print(
    f"{observed_difference:.6f}"
)


print("\nObserved difference in percentage points:")
print(
    f"{observed_difference * 100:.3f} pp"
)


# ============================================================
# 12. 95% PAIRED BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

RANDOM_SEED = 42
N_BOOTSTRAP = 10000

rng = np.random.default_rng(
    RANDOM_SEED
)


differences = (
    claim_scores["difference"]
    .to_numpy(
        dtype=float
    )
)


bootstrap_means = np.empty(
    N_BOOTSTRAP
)


n_claims = len(
    differences
)


for i in range(
    N_BOOTSTRAP
):

    sample = rng.choice(
        differences,
        size=n_claims,
        replace=True
    )

    bootstrap_means[i] = (
        sample.mean()
    )


ci_lower = np.percentile(
    bootstrap_means,
    2.5
)

ci_upper = np.percentile(
    bootstrap_means,
    97.5
)


print("\n95% bootstrap CI:")
print(
    f"[{ci_lower:.6f}, "
    f"{ci_upper:.6f}]"
)


print("\n95% bootstrap CI in percentage points:")
print(
    f"[{ci_lower * 100:.3f}, "
    f"{ci_upper * 100:.3f}] pp"
)


# ============================================================
# 13. PAIRED PERMUTATION TEST
# ============================================================

# Under H0, the labels "evidence present" and
# "evidence absent" are exchangeable within each claim.
#
# Swapping the two conditions is equivalent to randomly
# multiplying each paired difference by +1 or -1.

N_PERMUTATIONS = 10000

permutation_means = np.empty(
    N_PERMUTATIONS
)


for i in range(
    N_PERMUTATIONS
):

    signs = rng.choice(
        [-1, 1],
        size=n_claims
    )

    permutation_means[i] = (
        (
            differences
            * signs
        ).mean()
    )


# Two-sided p-value.
#
# +1 correction prevents an estimated p-value of exactly zero.

extreme_count = np.sum(
    np.abs(
        permutation_means
    )
    >=
    abs(
        observed_difference
    )
)


p_value = (
    extreme_count + 1
) / (
    N_PERMUTATIONS + 1
)


print("\nTwo-sided paired permutation p-value:")
print(
    f"{p_value:.6f}"
)


# ============================================================
# 14. DIRECTION OF CLAIM-LEVEL DIFFERENCES
# ============================================================

more_sensitive_without = int(
    (
        differences > 0
    ).sum()
)

same_sensitivity = int(
    (
        differences == 0
    ).sum()
)

more_sensitive_with = int(
    (
        differences < 0
    ).sum()
)


print("\nClaim-level direction:")

print(
    "Higher FCR without evidence:",
    more_sensitive_without
)

print(
    "Same FCR:",
    same_sensitivity
)

print(
    "Higher FCR with evidence:",
    more_sensitive_with
)


print("\n" + "=" * 70)
print("PRIMARY H2 INFERENCE COMPLETE")
print("=" * 70)
# ============================================================
# 15. H1: FCR CONFIDENCE INTERVALS
# ============================================================

print("\n" + "=" * 70)
print("H1: FRAMING SENSITIVITY")
print("=" * 70)

N_BOOTSTRAP_H1 = 10000

rng_h1 = np.random.default_rng(42)

h1_rows = []


for evidence_condition in EVIDENCE_CONDITIONS:

    print(
        f"\nEVIDENCE {evidence_condition.upper()}"
    )

    for emotion in EMOTIONS:

        subset = comparisons[
            (
                comparisons["evidence_condition"]
                == evidence_condition
            )
            &
            (
                comparisons["emotion"]
                == emotion
            )
            &
            (
                comparisons["valid_pair"]
            )
        ].copy()


        values = (
            subset["framing_change"]
            .astype(int)
            .to_numpy()
        )


        observed_fcr = values.mean()


        bootstrap_fcr = np.empty(
            N_BOOTSTRAP_H1
        )


        for i in range(
            N_BOOTSTRAP_H1
        ):

            sample = rng_h1.choice(
                values,
                size=len(values),
                replace=True
            )

            bootstrap_fcr[i] = (
                sample.mean()
            )


        ci_lower = np.percentile(
            bootstrap_fcr,
            2.5
        )

        ci_upper = np.percentile(
            bootstrap_fcr,
            97.5
        )


        h1_rows.append(
            {
                "evidence_condition":
                    evidence_condition,

                "emotion":
                    emotion,

                "n":
                    len(values),

                "fcr":
                    observed_fcr,

                "ci_lower":
                    ci_lower,

                "ci_upper":
                    ci_upper,
            }
        )


        print(
            f"{emotion:8s} "
            f"FCR = {observed_fcr * 100:.3f}%   "
            f"95% CI "
            f"[{ci_lower * 100:.3f}%, "
            f"{ci_upper * 100:.3f}%]"
        )


h1_table = pd.DataFrame(
    h1_rows
)


# ============================================================
# 16. H3: DO EMOTION TYPES DIFFER?
# Repeated-measures permutation test
# ============================================================

print("\n" + "=" * 70)
print("H3: DIFFERENCES BETWEEN EMOTION TYPES")
print("=" * 70)


N_PERMUTATIONS_H3 = 10000

rng_h3 = np.random.default_rng(42)

h3_rows = []


for evidence_condition in EVIDENCE_CONDITIONS:

    condition_data = comparisons[
        (
            comparisons["evidence_condition"]
            == evidence_condition
        )
        &
        (
            comparisons["id"].isin(
                complete_ids
            )
        )
    ].copy()


    matrix = (
        condition_data
        .pivot(
            index="id",
            columns="emotion",
            values="framing_change"
        )[EMOTIONS]
        .astype(int)
        .to_numpy()
    )


    observed_rates = (
        matrix.mean(axis=0)
    )


    # Statistic:
    # variance across the four emotion-specific FCRs.
    #
    # Under H0, emotion labels are exchangeable
    # within each claim.

    observed_statistic = (
        np.var(
            observed_rates
        )
    )


    permutation_statistics = np.empty(
        N_PERMUTATIONS_H3
    )


    for i in range(
        N_PERMUTATIONS_H3
    ):

        permuted = np.empty_like(
            matrix
        )


        for row_index in range(
            matrix.shape[0]
        ):

            permuted[
                row_index
            ] = rng_h3.permutation(
                matrix[row_index]
            )


        permuted_rates = (
            permuted.mean(axis=0)
        )


        permutation_statistics[i] = (
            np.var(
                permuted_rates
            )
        )


    extreme_count = np.sum(
        permutation_statistics
        >= observed_statistic
    )


    p_value_h3 = (
        extreme_count + 1
    ) / (
        N_PERMUTATIONS_H3 + 1
    )


    print(
        f"\nEVIDENCE {evidence_condition.upper()}"
    )


    for emotion, rate in zip(
        EMOTIONS,
        observed_rates
    ):

        print(
            f"{emotion:8s}: "
            f"{rate * 100:.3f}%"
        )


    print(
        "Omnibus permutation p-value:",
        f"{p_value_h3:.6f}"
    )


    h3_rows.append(
        {
            "evidence_condition":
                evidence_condition,

            "p_value":
                p_value_h3,

            "observed_variance":
                observed_statistic,
        }
    )


h3_table = pd.DataFrame(
    h3_rows
)


# ============================================================
# 17. SAVE H1 AND H3 RESULTS
# ============================================================

h1_table.to_csv(
    OUTPUT_DIR
    / "h1_fcr_confidence_intervals.csv",
    index=False
)


h3_table.to_csv(
    OUTPUT_DIR
    / "h3_emotion_test.csv",
    index=False
)


print("\n" + "=" * 70)
print("H1 AND H3 ANALYSIS COMPLETE")
print("=" * 70)

# ============================================================
# 18. H4: DOES SENSITIVITY DIFFER BY GROUND-TRUTH LABEL?
# ============================================================

print("\n" + "=" * 70)
print("H4: FRAMING SENSITIVITY BY GROUND-TRUTH LABEL")
print("=" * 70)


# Use the same 1499 complete claims as the primary analysis.

complete_claim_fcr = (
    complete_comparisons
    .groupby(
        [
            "id",
            "label",
            "evidence_condition",
        ]
    )["framing_change"]
    .mean()
    .reset_index()
)


N_BOOTSTRAP_H4 = 10000
rng_h4 = np.random.default_rng(42)

h4_rows = []


for evidence_condition in EVIDENCE_CONDITIONS:

    condition_scores = complete_claim_fcr[
        complete_claim_fcr["evidence_condition"]
        == evidence_condition
    ]


    supports = (
        condition_scores[
            condition_scores["label"] == "SUPPORTS"
        ]["framing_change"]
        .to_numpy(dtype=float)
    )


    refutes = (
        condition_scores[
            condition_scores["label"] == "REFUTES"
        ]["framing_change"]
        .to_numpy(dtype=float)
    )


    supports_mean = supports.mean()
    refutes_mean = refutes.mean()

    observed_difference = (
        supports_mean - refutes_mean
    )


    # --------------------------------------------------------
    # Bootstrap CI for SUPPORTS - REFUTES
    # --------------------------------------------------------

    bootstrap_differences = np.empty(
        N_BOOTSTRAP_H4
    )


    for i in range(N_BOOTSTRAP_H4):

        supports_sample = rng_h4.choice(
            supports,
            size=len(supports),
            replace=True
        )

        refutes_sample = rng_h4.choice(
            refutes,
            size=len(refutes),
            replace=True
        )

        bootstrap_differences[i] = (
            supports_sample.mean()
            -
            refutes_sample.mean()
        )


    ci_lower = np.percentile(
        bootstrap_differences,
        2.5
    )

    ci_upper = np.percentile(
        bootstrap_differences,
        97.5
    )


    # --------------------------------------------------------
    # Permutation test for label difference
    # --------------------------------------------------------

    combined = np.concatenate(
        [supports, refutes]
    )

    n_supports = len(supports)

    N_PERMUTATIONS_H4 = 10000

    permutation_differences = np.empty(
        N_PERMUTATIONS_H4
    )


    for i in range(N_PERMUTATIONS_H4):

        shuffled = rng_h4.permutation(
            combined
        )

        perm_supports = shuffled[
            :n_supports
        ]

        perm_refutes = shuffled[
            n_supports:
        ]

        permutation_differences[i] = (
            perm_supports.mean()
            -
            perm_refutes.mean()
        )


    extreme_count = np.sum(
        np.abs(permutation_differences)
        >=
        abs(observed_difference)
    )


    p_value = (
        extreme_count + 1
    ) / (
        N_PERMUTATIONS_H4 + 1
    )


    print(
        f"\nEVIDENCE {evidence_condition.upper()}"
    )

    print(
        "SUPPORTS FCR:",
        f"{supports_mean * 100:.3f}%"
    )

    print(
        "REFUTES FCR:",
        f"{refutes_mean * 100:.3f}%"
    )

    print(
        "Difference SUPPORTS - REFUTES:",
        f"{observed_difference * 100:.3f} pp"
    )

    print(
        "95% bootstrap CI:",
        f"[{ci_lower * 100:.3f}, "
        f"{ci_upper * 100:.3f}] pp"
    )

    print(
        "Two-sided permutation p-value:",
        f"{p_value:.6f}"
    )


    h4_rows.append(
        {
            "evidence_condition":
                evidence_condition,

            "supports_fcr":
                supports_mean,

            "refutes_fcr":
                refutes_mean,

            "difference":
                observed_difference,

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            "p_value":
                p_value,
        }
    )


h4_table = pd.DataFrame(
    h4_rows
)


# ============================================================
# 19. ACCURACY: EVIDENCE PRESENT VS ABSENT
# ============================================================

print("\n" + "=" * 70)
print("SECONDARY ANALYSIS: ACCURACY EFFECT OF EVIDENCE")
print("=" * 70)


accuracy_inference_rows = []


for emotion in [
    "neutral",
    "positive",
    "negative",
    "fear",
    "anger",
]:

    emotion_data = df[
        df["emotion"] == emotion
    ][
        [
            "id",
            "label",
            "evidence_condition",
            "prediction",
        ]
    ].copy()

    emotion_data["correct"] = (
        emotion_data["prediction"]
        ==
        emotion_data["label"]
    ).astype("boolean")

    # Preserve the one technical missing response
    # as missing rather than counting it as incorrect.
    emotion_data.loc[
        emotion_data["prediction"].isna(),
        "correct"
    ] = pd.NA

    accuracy_wide = (
        emotion_data
        .pivot(
            index="id",
            columns="evidence_condition",
            values="correct"
        )
    )

    accuracy_wide = (
        accuracy_wide.dropna(
            subset=[
                "present",
                "absent",
            ]
        )
    )

    present_correct = (
        accuracy_wide["present"]
        .astype(int)
        .to_numpy()
    )

    absent_correct = (
        accuracy_wide["absent"]
        .astype(int)
        .to_numpy()
    )


    present_accuracy = (
        present_correct.mean()
    )

    absent_accuracy = (
        absent_correct.mean()
    )

    accuracy_difference = (
        present_accuracy
        -
        absent_accuracy
    )


    # --------------------------------------------------------
    # Paired bootstrap CI
    # --------------------------------------------------------

    paired_differences = (
        present_correct
        -
        absent_correct
    )


    rng_accuracy = np.random.default_rng(
        42
    )

    N_BOOTSTRAP_ACCURACY = 10000

    bootstrap_accuracy = np.empty(
        N_BOOTSTRAP_ACCURACY
    )


    for i in range(
        N_BOOTSTRAP_ACCURACY
    ):

        sample = rng_accuracy.choice(
            paired_differences,
            size=len(paired_differences),
            replace=True
        )

        bootstrap_accuracy[i] = (
            sample.mean()
        )


    ci_lower = np.percentile(
        bootstrap_accuracy,
        2.5
    )

    ci_upper = np.percentile(
        bootstrap_accuracy,
        97.5
    )


    # --------------------------------------------------------
    # Exact paired sign-flip permutation test
    # via Monte Carlo
    # --------------------------------------------------------

    rng_permutation = np.random.default_rng(
        42
    )

    N_PERMUTATIONS_ACCURACY = 10000

    permutation_means = np.empty(
        N_PERMUTATIONS_ACCURACY
    )


    for i in range(
        N_PERMUTATIONS_ACCURACY
    ):

        signs = rng_permutation.choice(
            [-1, 1],
            size=len(paired_differences)
        )

        permutation_means[i] = (
            (
                paired_differences
                * signs
            ).mean()
        )


    extreme_count = np.sum(
        np.abs(permutation_means)
        >=
        abs(accuracy_difference)
    )


    p_value = (
        extreme_count + 1
    ) / (
        N_PERMUTATIONS_ACCURACY + 1
    )


    # --------------------------------------------------------
    # Discordant pairs
    # --------------------------------------------------------

    evidence_helps = int(
        (
            (present_correct == 1)
            &
            (absent_correct == 0)
        ).sum()
    )

    evidence_hurts = int(
        (
            (present_correct == 0)
            &
            (absent_correct == 1)
        ).sum()
    )


    print(
        f"\n{emotion.upper()}"
    )

    print(
        "N paired claims:",
        len(paired_differences)
    )

    print(
        "Accuracy with evidence:",
        f"{present_accuracy * 100:.3f}%"
    )

    print(
        "Accuracy without evidence:",
        f"{absent_accuracy * 100:.3f}%"
    )

    print(
        "Difference present - absent:",
        f"{accuracy_difference * 100:.3f} pp"
    )

    print(
        "95% bootstrap CI:",
        f"[{ci_lower * 100:.3f}, "
        f"{ci_upper * 100:.3f}] pp"
    )

    print(
        "Two-sided paired permutation p-value:",
        f"{p_value:.6f}"
    )

    print(
        "Evidence helps / hurts:",
        f"{evidence_helps} / {evidence_hurts}"
    )


    accuracy_inference_rows.append(
        {
            "emotion":
                emotion,

            "n":
                len(paired_differences),

            "accuracy_present":
                present_accuracy,

            "accuracy_absent":
                absent_accuracy,

            "difference":
                accuracy_difference,

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            "p_value":
                p_value,

            "evidence_helps":
                evidence_helps,

            "evidence_hurts":
                evidence_hurts,
        }
    )


accuracy_inference_table = pd.DataFrame(
    accuracy_inference_rows
)


# ============================================================
# 20. SAVE FINAL STATISTICAL TABLES
# ============================================================

h4_table.to_csv(
    OUTPUT_DIR
    / "h4_label_sensitivity.csv",
    index=False
)


accuracy_inference_table.to_csv(
    OUTPUT_DIR
    / "accuracy_inference.csv",
    index=False
)


print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 70)