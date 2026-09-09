import pandas as pd


INPUT_FILE = "data/FEVER/fever_experiment_v2.csv"

OUTPUT_FILE = "data/FEVER/fever_final_sample.csv"

SAMPLE_PER_LABEL = 750
RANDOM_SEED = 42

LABELS = [
    "SUPPORTS",
    "REFUTES"
]


df = pd.read_csv(INPUT_FILE)

print("\nSource dataset:")
print(INPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nLabel counts:")
print(df["label"].value_counts())



required_columns = {
    "id",
    "claim",
    "label",
    "evidence_text"
}

missing_columns = (
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


if df["id"].duplicated().any():
    raise ValueError(
        "Duplicate FEVER IDs found."
    )


if df["claim"].duplicated().any():
    raise ValueError(
        "Duplicate claims found."
    )


unexpected_labels = (
    set(df["label"].unique())
    - set(LABELS)
)

if unexpected_labels:
    raise ValueError(
        f"Unexpected labels: {unexpected_labels}"
    )


samples = []


for label in LABELS:

    label_df = df[
        df["label"] == label
    ].copy()


    if len(label_df) < SAMPLE_PER_LABEL:
        raise ValueError(
            f"Not enough {label} examples."
        )


    sampled = label_df.sample(
        n=SAMPLE_PER_LABEL,
        random_state=RANDOM_SEED
    )


    samples.append(sampled)


final_sample = pd.concat(
    samples,
    ignore_index=True
)


final_sample = (
    final_sample
    .sample(
        frac=1,
        random_state=RANDOM_SEED
    )
    .reset_index(drop=True)
)


expected_total = (
    SAMPLE_PER_LABEL
    * len(LABELS)
)


if len(final_sample) != expected_total:
    raise ValueError(
        "Unexpected final sample size."
    )


label_counts = (
    final_sample["label"]
    .value_counts()
)


for label in LABELS:

    if label_counts[label] != SAMPLE_PER_LABEL:

        raise ValueError(
            f"Expected {SAMPLE_PER_LABEL} "
            f"{label} examples."
        )


if final_sample["id"].duplicated().any():
    raise ValueError(
        "Duplicate IDs in final sample."
    )


if final_sample["claim"].duplicated().any():
    raise ValueError(
        "Duplicate claims in final sample."
    )


if final_sample["evidence_text"].isna().any():
    raise ValueError(
        "Missing evidence found."
    )


final_sample.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFinal shape:")
print(final_sample.shape)

print("\nLabel counts:")
print(
    final_sample["label"]
    .value_counts()
)

print("\nUnique IDs:")
print(
    final_sample["id"].nunique()
)

print("\nUnique claims:")
print(
    final_sample["claim"].nunique()
)


if "num_evidence_sentences" in final_sample.columns:

    print("\nEvidence sentence counts:")

    print(
        final_sample[
            "num_evidence_sentences"
        ]
        .value_counts()
        .sort_index()
    )


print("\nRandom seed:")
print(RANDOM_SEED)

print("\nSaved to:")
print(OUTPUT_FILE)
