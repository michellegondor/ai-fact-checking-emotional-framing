import json
from pathlib import Path

import pandas as pd



RESULT_FILE = Path(
    "data/FEVER/final_batch/batch_results.jsonl"
)

METADATA_FILE = Path(
    "data/FEVER/final_batch/metadata.csv"
)

OUTPUT_FILE = Path(
    "data/FEVER/final_batch/final_results.csv"
)

DIAGNOSTICS_FILE = Path(
    "data/FEVER/final_batch/parse_diagnostics.csv"
)

EXPECTED_ROWS = 15000

VALID_LABELS = {
    "SUPPORTS",
    "REFUTES",
}



def extract_classification(response):

    candidates = response.get(
        "candidates",
        []
    )

    if not candidates:
        return None, None, "no_candidates"

    valid_outputs = []
    seen_texts = []

    for candidate in candidates:

        content = candidate.get(
            "content",
            {}
        )

        parts = content.get(
            "parts",
            []
        )

        for part in parts:

            text = part.get("text")

            if not text:
                continue

            text = text.strip()

            if not text:
                continue

            seen_texts.append(text)

            if part.get("thought") is True:
                continue

            try:
                parsed = json.loads(text)

            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            prediction = parsed.get(
                "classification"
            )

            if prediction in VALID_LABELS:

                valid_outputs.append(
                    (
                        prediction,
                        text
                    )
                )

    if len(valid_outputs) == 0:

        return (
            None,
            " | ".join(seen_texts),
            "no_valid_classification"
        )

    unique_predictions = {
        prediction
        for prediction, _ in valid_outputs
    }

    if len(unique_predictions) > 1:

        return (
            None,
            " | ".join(
                text
                for _, text in valid_outputs
            ),
            "conflicting_classifications"
        )

    return (
        valid_outputs[0][0],
        valid_outputs[0][1],
        None
    )



metadata = pd.read_csv(
    METADATA_FILE
)


if len(metadata) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} metadata rows, "
        f"found {len(metadata)}."
    )


if metadata[
    "request_key"
].duplicated().any():

    raise ValueError(
        "Duplicate metadata request keys."
    )


print(
    "\nMetadata rows:",
    len(metadata)
)


parsed_rows = []
diagnostic_rows = []


with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line_number, line in enumerate(
        f,
        start=1
    ):

        if not line.strip():
            continue

        obj = json.loads(line)

        request_key = obj.get(
            "key"
        )


        if request_key is None:

            raise ValueError(
                f"Missing request key "
                f"on line {line_number}."
            )


        error = obj.get(
            "error"
        )


        if error is not None:

            parsed_rows.append(
                {
                    "request_key":
                        request_key,

                    "prediction":
                        None,

                    "raw_response":
                        None,

                    "missing_prediction":
                        True,

                    "missing_reason":
                        "api_error",
                }
            )

            diagnostic_rows.append(
                {
                    "line_number":
                        line_number,

                    "request_key":
                        request_key,

                    "reason":
                        "api_error",

                    "finish_reason":
                        None,

                    "observed_text":
                        str(error),
                }
            )

            continue


        response = obj.get(
            "response"
        )


        if response is None:

            parsed_rows.append(
                {
                    "request_key":
                        request_key,

                    "prediction":
                        None,

                    "raw_response":
                        None,

                    "missing_prediction":
                        True,

                    "missing_reason":
                        "missing_response",
                }
            )

            diagnostic_rows.append(
                {
                    "line_number":
                        line_number,

                    "request_key":
                        request_key,

                    "reason":
                        "missing_response",

                    "finish_reason":
                        None,

                    "observed_text":
                        None,
                }
            )

            continue


        candidates = response.get(
            "candidates",
            []
        )

        finish_reason = None

        if candidates:

            finish_reason = (
                candidates[0]
                .get("finishReason")
            )


        (
            prediction,
            raw_response,
            missing_reason,
        ) = extract_classification(
            response
        )


        is_missing = (
            prediction is None
        )


        parsed_rows.append(
            {
                "request_key":
                    request_key,

                "prediction":
                    prediction,

                "raw_response":
                    raw_response,

                "missing_prediction":
                    is_missing,

                "missing_reason":
                    missing_reason,
            }
        )


        if is_missing:

            diagnostic_rows.append(
                {
                    "line_number":
                        line_number,

                    "request_key":
                        request_key,

                    "reason":
                        missing_reason,

                    "finish_reason":
                        finish_reason,

                    "observed_text":
                        raw_response,
                }
            )


predictions = pd.DataFrame(
    parsed_rows
)

diagnostics = pd.DataFrame(
    diagnostic_rows
)


print(
    "\nBatch rows parsed:",
    len(predictions)
)


if len(predictions) != EXPECTED_ROWS:

    raise ValueError(
        f"Expected {EXPECTED_ROWS} response rows, "
        f"found {len(predictions)}."
    )


if predictions[
    "request_key"
].duplicated().any():

    raise ValueError(
        "Duplicate response keys."
    )



metadata_keys = set(
    metadata[
        "request_key"
    ]
)

prediction_keys = set(
    predictions[
        "request_key"
    ]
)


missing_keys = (
    metadata_keys
    - prediction_keys
)

unexpected_keys = (
    prediction_keys
    - metadata_keys
)


print(
    "Missing request keys:",
    len(missing_keys)
)

print(
    "Unexpected request keys:",
    len(unexpected_keys)
)


if missing_keys:
    raise ValueError(
        "Metadata contains requests "
        "with no batch response."
    )


if unexpected_keys:
    raise ValueError(
        "Batch contains unexpected "
        "request keys."
    )



valid_count = (
    predictions[
        "prediction"
    ]
    .notna()
    .sum()
)

missing_count = (
    predictions[
        "missing_prediction"
    ]
    .sum()
)


print(
    "\nValid classifications:",
    valid_count
)

print(
    "Missing classifications:",
    missing_count
)


if valid_count != 14999:
    raise ValueError(
        f"Expected 14999 valid classifications, "
        f"found {valid_count}."
    )


if missing_count != 1:
    raise ValueError(
        f"Expected exactly 1 missing "
        f"classification, found {missing_count}."
    )


missing_rows = predictions[
    predictions[
        "missing_prediction"
    ]
]


missing_keys_found = set(
    missing_rows[
        "request_key"
    ]
)


expected_missing_keys = {
    "217251__absent__anger"
}


if missing_keys_found != expected_missing_keys:

    raise ValueError(
        "The missing response is not the "
        "previously diagnosed technical case."
    )



diagnostics.to_csv(
    DIAGNOSTICS_FILE,
    index=False
)


results = metadata.merge(
    predictions,
    on="request_key",
    how="left",
    validate="one_to_one",
)


if len(results) != EXPECTED_ROWS:

    raise ValueError(
        "Merged dataset does not "
        "contain 15000 rows."
    )



results["correct"] = (
    results["prediction"]
    == results["label"]
).astype("boolean")


results.loc[
    results["missing_prediction"],
    "correct"
] = pd.NA


print(
    "\nPrediction counts:"
)

print(
    results[
        "prediction"
    ].value_counts(
        dropna=False
    )
)


print(
    "\nRows by evidence condition:"
)

print(
    results[
        "evidence_condition"
    ].value_counts()
)


print(
    "\nRows by emotion:"
)

print(
    results[
        "emotion"
    ].value_counts()
)


requests_per_claim = (
    results
    .groupby("id")
    .size()
)


if not (
    requests_per_claim == 10
).all():

    raise ValueError(
        "Not every claim has exactly "
        "10 experimental rows."
    )


if results[
    "request_key"
].nunique() != EXPECTED_ROWS:

    raise ValueError(
        "Final request keys are "
        "not unique."
    )



print(
    "\nAccuracy by evidence condition "
)


accuracy = (
    results
    .dropna(
        subset=["prediction"]
    )
    .groupby(
        "evidence_condition"
    )["correct"]
    .mean()
)


print(
    accuracy
)


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nExperimental rows:",
    len(results)
)

print(
    "Valid classifications:",
    valid_count
)

print(
    "Technical missing:",
    missing_count
)

print(
    "Unique claims:",
    results[
        "id"
    ].nunique()
)

print(
    "Unique request keys:",
    results[
        "request_key"
    ].nunique()
)

print(
    "\nFinal dataset saved to:"
)

print(
    OUTPUT_FILE
)
