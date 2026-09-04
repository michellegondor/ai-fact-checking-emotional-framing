import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_FILE = "data/FEVER/fever_final_sample.csv"
FRAMED_FILE = "data/FEVER/fever_framed_v2.csv"

OUTPUT_DIR = Path("data/FEVER/final_batch")

REQUESTS_FILE = OUTPUT_DIR / "requests.jsonl"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

EXPECTED_CLAIMS = 1500

EMOTION_ORDER = [
    "neutral",
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
# FINAL FROZEN PROMPTS
# ============================================================

PROMPT_WITH_EVIDENCE = """
Evaluate the factual claim based on the provided evidence.

Evidence:
{evidence}

Statement:
{text}

Classify the factual claim as exactly one of:

SUPPORTS
REFUTES

Return only the classification.
""".strip()


PROMPT_NO_EVIDENCE = """
Evaluate the factual claim in the following statement.

Statement:
{text}

Classify the factual claim as exactly one of:

SUPPORTS
REFUTES

Return only the classification.
""".strip()


# ============================================================
# 1. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD FROZEN SAMPLE
# ============================================================

sample = pd.read_csv(
    SAMPLE_FILE
)


print("=" * 70)
print("PREPARING FINAL BATCH")
print("=" * 70)

print("\nFrozen sample:")
print(SAMPLE_FILE)

print("\nClaims:")
print(len(sample))


if len(sample) != EXPECTED_CLAIMS:
    raise ValueError(
        f"Expected {EXPECTED_CLAIMS} claims, "
        f"found {len(sample)}."
    )


if sample["id"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found in frozen sample."
    )


if sample["claim"].duplicated().any():
    raise ValueError(
        "Duplicate claims found in frozen sample."
    )


label_counts = (
    sample["label"]
    .value_counts()
)


print("\nLabel counts:")
print(label_counts)


if label_counts.get("SUPPORTS", 0) != 750:
    raise ValueError(
        "Expected exactly 750 SUPPORTS."
    )


if label_counts.get("REFUTES", 0) != 750:
    raise ValueError(
        "Expected exactly 750 REFUTES."
    )


# ============================================================
# 3. LOAD FROZEN FRAMINGS
# ============================================================

framed = pd.read_csv(
    FRAMED_FILE
)


final_ids = set(
    sample["id"].astype(int)
)


framed = framed[
    framed["id"]
    .astype(int)
    .isin(final_ids)
].copy()


expected_framed_rows = (
    EXPECTED_CLAIMS
    * len(EMOTION_ORDER)
)


print(
    "\nFramed rows:",
    len(framed)
)


if len(framed) != expected_framed_rows:
    raise ValueError(
        f"Expected {expected_framed_rows} "
        f"framed rows, found {len(framed)}."
    )


# ============================================================
# 4. VALIDATE FIVE CONDITIONS PER CLAIM
# ============================================================

rows_per_claim = (
    framed
    .groupby("id")
    .size()
)


if not (rows_per_claim == 5).all():
    raise ValueError(
        "Not every claim has exactly "
        "5 framing conditions."
    )


emotion_sets = (
    framed
    .groupby("id")["emotion"]
    .apply(set)
)


expected_emotions = set(
    EMOTION_ORDER
)


if not emotion_sets.apply(
    lambda x: x == expected_emotions
).all():

    raise ValueError(
        "At least one claim does not contain "
        "all five framing conditions."
    )


# ============================================================
# 5. VALIDATE CLAIM TEXT
# ============================================================

sample_claims = (
    sample
    .set_index("id")["claim"]
    .astype(str)
)


neutral_rows = framed[
    framed["emotion"] == "neutral"
]


for _, row in neutral_rows.iterrows():

    claim_id = int(row["id"])

    expected_claim = (
        sample_claims.loc[claim_id]
    )

    actual_text = str(
        row["text"]
    )

    if actual_text != expected_claim:

        raise ValueError(
            f"Neutral text changed for "
            f"claim ID {claim_id}."
        )


print(
    "\nNeutral claims match frozen "
    "original claims exactly."
)


# ============================================================
# 6. SORT DETERMINISTICALLY
# ============================================================

framed["emotion"] = pd.Categorical(
    framed["emotion"],
    categories=EMOTION_ORDER,
    ordered=True
)


framed = (
    framed
    .sort_values(
        ["id", "emotion"]
    )
    .reset_index(drop=True)
)


sample_lookup = (
    sample
    .set_index("id")
)


# ============================================================
# 7. BUILD REQUESTS + METADATA
# ============================================================

request_rows = []
metadata_rows = []


for _, row in framed.iterrows():

    claim_id = int(row["id"])

    label = str(
        row["label"]
    )

    emotion = str(
        row["emotion"]
    )

    template_id = str(
        row["template_id"]
    )

    text = str(
        row["text"]
    )

    original_claim = str(
        sample_lookup.loc[
            claim_id,
            "claim"
        ]
    )

    evidence = str(
        sample_lookup.loc[
            claim_id,
            "evidence_text"
        ]
    )


    for evidence_condition in EVIDENCE_CONDITIONS:

        request_key = (
            f"{claim_id}__"
            f"{evidence_condition}__"
            f"{emotion}"
        )


        if evidence_condition == "present":

            prompt = (
                PROMPT_WITH_EVIDENCE
                .format(
                    evidence=evidence,
                    text=text
                )
            )

        else:

            prompt = (
                PROMPT_NO_EVIDENCE
                .format(
                    text=text
                )
            )


        # JSONL request format.
        #
        # "key" lets us reconnect the response
        # with the exact experimental condition.
        request = {
            "key": request_key,

            "request": {
                "contents": [
                    {
                        "role": "user",

                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],

                "generationConfig": {
                    "responseMimeType":
                        "application/json",

                    "responseJsonSchema": {
                        "type": "object",

                        "properties": {
                            "classification": {
                                "type": "string",

                                "enum": [
                                    "SUPPORTS",
                                    "REFUTES"
                                ]
                            }
                        },

                        "required": [
                            "classification"
                        ],

                        "additionalProperties":
                            False
                    }
                }
            }
        }


        request_rows.append(
            request
        )


        metadata_rows.append(
            {
                "request_key":
                    request_key,

                "id":
                    claim_id,

                "original_claim":
                    original_claim,

                "label":
                    label,

                "emotion":
                    emotion,

                "evidence_condition":
                    evidence_condition,

                "template_id":
                    template_id,

                "text":
                    text,

                "num_evidence_sentences":
                    sample_lookup.loc[
                        claim_id,
                        "num_evidence_sentences"
                    ],

                "num_evidence_sets":
                    sample_lookup.loc[
                        claim_id,
                        "num_evidence_sets"
                    ],
            }
        )


# ============================================================
# 8. GLOBAL VALIDATION
# ============================================================

expected_requests = (
    EXPECTED_CLAIMS
    * len(EMOTION_ORDER)
    * len(EVIDENCE_CONDITIONS)
)


print(
    "\nRequests constructed:",
    len(request_rows)
)


if len(request_rows) != expected_requests:

    raise ValueError(
        f"Expected {expected_requests} requests, "
        f"got {len(request_rows)}."
    )


metadata = pd.DataFrame(
    metadata_rows
)


if len(metadata) != expected_requests:

    raise ValueError(
        "Metadata count mismatch."
    )


# ============================================================
# 9. UNIQUE REQUEST KEYS
# ============================================================

if metadata[
    "request_key"
].duplicated().any():

    duplicates = metadata[
        metadata[
            "request_key"
        ].duplicated(
            keep=False
        )
    ]

    print(duplicates)

    raise ValueError(
        "Duplicate request keys found."
    )


print(
    "All request keys are unique."
)


# ============================================================
# 10. CONDITION COUNTS
# ============================================================

print(
    "\nEvidence-condition counts:"
)

print(
    metadata[
        "evidence_condition"
    ].value_counts()
)


print(
    "\nEmotion counts:"
)

print(
    metadata[
        "emotion"
    ].value_counts()
)


print(
    "\nLabel counts across requests:"
)

print(
    metadata[
        "label"
    ].value_counts()
)


condition_table = pd.crosstab(
    metadata["emotion"],
    metadata["evidence_condition"]
)


print(
    "\nEmotion × evidence table:"
)

print(
    condition_table
)


expected_per_cell = EXPECTED_CLAIMS


if not (
    condition_table
    == expected_per_cell
).all().all():

    raise ValueError(
        "Experimental cells are not balanced."
    )


# ============================================================
# 11. CHECK 10 REQUESTS PER CLAIM
# ============================================================

requests_per_claim = (
    metadata
    .groupby("id")
    .size()
)


if not (
    requests_per_claim == 10
).all():

    raise ValueError(
        "Not every claim has exactly "
        "10 requests."
    )


print(
    "\nEvery claim has exactly "
    "10 experimental conditions."
)


# ============================================================
# 12. WRITE JSONL
# ============================================================

with open(
    REQUESTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    for request in request_rows:

        f.write(
            json.dumps(
                request,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# 13. WRITE METADATA
# ============================================================

metadata.to_csv(
    METADATA_FILE,
    index=False
)


# ============================================================
# 14. VERIFY WRITTEN JSONL
# ============================================================

line_count = 0
written_keys = set()


with open(
    REQUESTS_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line_count += 1

        parsed = json.loads(
            line
        )

        written_keys.add(
            parsed["key"]
        )


if line_count != expected_requests:

    raise ValueError(
        "Written JSONL does not contain "
        "exactly 15,000 lines."
    )


if len(written_keys) != expected_requests:

    raise ValueError(
        "Written JSONL request keys "
        "are not unique."
    )


# ============================================================
# 15. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL BATCH PREPARATION PASSED")
print("=" * 70)

print(
    "\nTotal claims:",
    EXPECTED_CLAIMS
)

print(
    "Conditions per claim:",
    10
)

print(
    "Total requests:",
    expected_requests
)

print(
    "\nJSONL lines:",
    line_count
)

print(
    "Unique request keys:",
    len(written_keys)
)

print(
    "\nRequests saved to:"
)

print(
    REQUESTS_FILE
)

print(
    "\nMetadata saved to:"
)

print(
    METADATA_FILE
)

print("\nNO API REQUESTS WERE SENT.")

print("=" * 70)