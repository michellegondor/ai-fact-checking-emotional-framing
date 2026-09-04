import json
from pathlib import Path

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

JOB_INFO_FILE = Path(
    "data/FEVER/final_batch/batch_job.json"
)

RESULT_FILE = Path(
    "data/FEVER/final_batch/batch_results.jsonl"
)


# ============================================================
# 1. LOAD SAVED JOB INFORMATION
# ============================================================

if not JOB_INFO_FILE.exists():
    raise FileNotFoundError(
        f"Job info file not found: {JOB_INFO_FILE}"
    )


with open(
    JOB_INFO_FILE,
    "r",
    encoding="utf-8"
) as f:

    job_info = json.load(f)


job_name = job_info["batch_job_name"]


print("=" * 70)
print("FINAL BATCH STATUS")
print("=" * 70)

print("\nBatch job:")
print(job_name)


# ============================================================
# 2. CONNECT TO GEMINI
# ============================================================

client = genai.Client()


# ============================================================
# 3. GET CURRENT JOB STATE
# ============================================================

batch_job = client.batches.get(
    name=job_name
)


state = batch_job.state.name


print("\nCurrent state:")
print(state)


# ============================================================
# 4. HANDLE NON-FINISHED STATES
# ============================================================

if state in {
    "JOB_STATE_PENDING",
    "JOB_STATE_RUNNING",
}:

    print(
        "\nThe experiment is still being processed."
    )

    print(
        "Run this script again later."
    )

    raise SystemExit(0)


# ============================================================
# 5. HANDLE FAILED STATES
# ============================================================

if state == "JOB_STATE_FAILED":

    print("\nThe batch job FAILED.")

    print("\nError information:")
    print(batch_job.error)

    raise SystemExit(1)


if state == "JOB_STATE_CANCELLED":

    print(
        "\nThe batch job was CANCELLED."
    )

    raise SystemExit(1)


if state == "JOB_STATE_EXPIRED":

    print(
        "\nThe batch job EXPIRED."
    )

    raise SystemExit(1)


# ============================================================
# 6. SUCCESS
# ============================================================

if state != "JOB_STATE_SUCCEEDED":

    raise RuntimeError(
        f"Unexpected job state: {state}"
    )


print(
    "\nBatch completed successfully."
)


# ============================================================
# 7. FIND RESULT FILE
# ============================================================

if not batch_job.dest:
    raise RuntimeError(
        "Batch succeeded but no destination "
        "information was returned."
    )


result_file_name = (
    batch_job.dest.file_name
)


if not result_file_name:
    raise RuntimeError(
        "Batch succeeded but no result "
        "file name was returned."
    )


print("\nRemote result file:")
print(result_file_name)


# ============================================================
# 8. DOWNLOAD RESULT FILE
# ============================================================

print(
    "\nDownloading results..."
)


file_content = client.files.download(
    file=result_file_name
)


RESULT_FILE.write_bytes(
    file_content
)


print("\nSaved to:")
print(RESULT_FILE)


# ============================================================
# 9. BASIC RESULT VALIDATION
# ============================================================

line_count = 0
keys = set()
error_count = 0


with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        line_count += 1

        obj = json.loads(line)

        key = obj.get("key")

        if key is not None:
            keys.add(key)

        if obj.get("error") is not None:
            error_count += 1


print("\n" + "=" * 70)
print("DOWNLOADED RESULT CHECK")
print("=" * 70)

print(
    "\nJSONL responses:",
    line_count
)

print(
    "Unique response keys:",
    len(keys)
)

print(
    "Responses containing errors:",
    error_count
)


if line_count != 15000:

    print(
        "\nWARNING:"
        " Expected 15000 responses."
    )


if len(keys) != 15000:

    print(
        "WARNING:"
        " Expected 15000 unique keys."
    )


if error_count > 0:

    print(
        "WARNING:"
        " Some requests returned errors."
    )


if (
    line_count == 15000
    and len(keys) == 15000
    and error_count == 0
):

    print(
        "\nAll 15,000 batch responses "
        "were received successfully."
    )


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)