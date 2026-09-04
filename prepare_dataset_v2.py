import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/FEVER/fever_with_evidence.csv"
OUTPUT_FILE = "data/FEVER/fever_experiment_v2.csv"

SAMPLES_PER_LABEL = 2000

MAX_EVIDENCE_SENTENCES = 6

RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nLabels:")
print(df["label"].value_counts())


# ============================================================
# 2. REMOVE CLAIMS WITH CONFLICTING LABELS
# ============================================================

label_counts = (
    df.groupby("claim")["label"]
    .nunique()
)

conflicting_claims = label_counts[
    label_counts > 1
].index


print(
    "\nClaims with conflicting labels:",
    len(conflicting_claims)
)


df = df[
    ~df["claim"].isin(conflicting_claims)
].copy()


# ============================================================
# 3. REMOVE DUPLICATE CLAIMS
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["claim"],
    keep="first"
).copy()

removed_duplicates = (
    before_duplicates - len(df)
)


print(
    "Duplicate claims removed:",
    removed_duplicates
)


# ============================================================
# 4. REMOVE INVALID / EMPTY EVIDENCE
# ============================================================

before_evidence = len(df)


df = df[
    df["evidence_text"].notna()
].copy()

df = df[
    df["evidence_text"].str.strip() != ""
].copy()


removed_invalid_evidence = (
    before_evidence - len(df)
)


print(
    "Rows invalid/empty evidence:",
    removed_invalid_evidence
)


# ============================================================
# 5. LIMIT EVIDENCE LENGTH
# ============================================================

before_length_filter = len(df)


df = df[
    df["num_evidence_sentences"]
    <= MAX_EVIDENCE_SENTENCES
].copy()


removed_long_evidence = (
    before_length_filter - len(df)
)


print(
    f"Rows with >{MAX_EVIDENCE_SENTENCES} "
    f"evidence sentences removed:",
    removed_long_evidence
)


# ============================================================
# 6. CLEAN DATA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEAN DATA")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nLabels:")
print(df["label"].value_counts())

print("\nEvidence sentence counts:")
print(
    df["num_evidence_sentences"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. BALANCED SAMPLE
# ============================================================

supports = (
    df[df["label"] == "SUPPORTS"]
    .sample(
        n=SAMPLES_PER_LABEL,
        random_state=RANDOM_STATE
    )
)

refutes = (
    df[df["label"] == "REFUTES"]
    .sample(
        n=SAMPLES_PER_LABEL,
        random_state=RANDOM_STATE
    )
)


experiment = pd.concat(
    [supports, refutes],
    ignore_index=True
)


experiment = experiment.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# 8. KEEP FINAL COLUMNS
# ============================================================

experiment = experiment[
    [
        "id",
        "claim",
        "label",
        "evidence_text",
        "evidence_pages",
        "evidence_sentence_ids",
        "num_evidence_sentences",
        "num_evidence_sets"
    ]
]


# ============================================================
# 9. FINAL CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EXPERIMENT DATASET")
print("=" * 70)

print("\nShape:")
print(experiment.shape)

print("\nLabels:")
print(
    experiment["label"].value_counts()
)

print("\nEvidence sentence counts:")
print(
    experiment[
        "num_evidence_sentences"
    ]
    .value_counts()
    .sort_index()
)

print("\nMaximum evidence sentences:")
print(
    experiment[
        "num_evidence_sentences"
    ].max()
)

print("\nUnique claims:")
print(
    experiment["claim"].nunique()
)


# ============================================================
# 10. SAVE
# ============================================================

experiment.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)