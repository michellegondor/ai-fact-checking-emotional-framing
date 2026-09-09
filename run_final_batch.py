import json
from pathlib import Path

from google import genai
from google.genai import types


MODEL = "gemini-3.7-flash"

INPUT_FILE = Path(
    "data/FEVER/final_batch/requests.jsonl"
)

JOB_INFO_FILE = Path(
    "data/FEVER/final_batch/batch_job.json"
)

DISPLAY_NAME = (
    "fever-emotional-framing-final"
)



if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


line_count = 0
keys = set()


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line_count += 1

        obj = json.loads(line)

        request_key = obj["key"]

        if request_key in keys:
            raise ValueError(
                f"Duplicate request key: "
                f"{request_key}"
            )

        keys.add(request_key)


if line_count != 15000:
    raise ValueError(
        f"Expected 15000 requests, "
        f"found {line_count}."
    )


if len(keys) != 15000:
    raise ValueError(
        "Request keys are not unique."
    )


print("\nLocal validation passed.")
print("Requests:", line_count)
print("Unique keys:", len(keys))

client = genai.Client()


uploaded_file = client.files.upload(
    file=str(INPUT_FILE),

    config=types.UploadFileConfig(
        display_name=(
            "fever-final-batch-requests"
        ),
        mime_type="jsonl",
    ),
)


print(
    "Uploaded file:",
    uploaded_file.name
)


batch_job = client.batches.create(
    model=MODEL,

    src=uploaded_file.name,

    config={
        "display_name":
            DISPLAY_NAME
    },
)


print(
    "\nBatch job created:",
    batch_job.name
)

print(
    "Initial state:",
    batch_job.state.name
)

job_info = {
    "model":
        MODEL,

    "input_file":
        str(INPUT_FILE),

    "uploaded_file_name":
        uploaded_file.name,

    "batch_job_name":
        batch_job.name,

    "display_name":
        DISPLAY_NAME,

    "num_requests":
        line_count,
}


with open(
    JOB_INFO_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        job_info,
        f,
        indent=2
    )


print(
    "\nJob information saved to:"
)

print(
    JOB_INFO_FILE
)
