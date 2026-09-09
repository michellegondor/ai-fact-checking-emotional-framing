import json
import os
import pandas as pd



TRAIN_FILE = "data/FEVER/train.jsonl"
WIKI_FOLDER = "data/FEVER/wiki-pages/wiki-pages"
OUTPUT_FILE = "data/FEVER/fever_with_evidence.csv"


wiki_lookup = {}

file_count = 0
page_count = 0


for filename in os.listdir(WIKI_FOLDER):

    if not filename.endswith(".jsonl"):
        continue

    file_path = os.path.join(
        WIKI_FOLDER,
        filename
    )

    file_count += 1

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            page = json.loads(line)

            page_id = page["id"]

            sentences = {}

            for sentence_line in (
                page.get("lines", "").split("\n")
            ):

                if not sentence_line.strip():
                    continue

                parts = sentence_line.split("\t")

                if len(parts) < 2:
                    continue

                try:
                    sentence_id = int(parts[0])

                except ValueError:
                    continue

                sentence_text = parts[1].strip()

                if sentence_text:
                    sentences[sentence_id] = sentence_text

            wiki_lookup[page_id] = sentences

            page_count += 1


print(f"Wikipedia files loaded: {file_count}")
print(f"Wikipedia pages loaded: {page_count:,}")


rows = []

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        rows.append(json.loads(line))


print(f"Total FEVER rows: {len(rows):,}")

output_rows = []

missing_complete_evidence = 0
kept_rows = 0


for row in rows:

    label = row["label"]

    if label not in {"SUPPORTS", "REFUTES"}:
        continue


    evidence_sets = row.get("evidence", [])


    valid_evidence_sets = []


    for evidence_set in evidence_sets:

        current_set = []

        set_is_complete = True


        for evidence_item in evidence_set:

            if len(evidence_item) < 4:

                set_is_complete = False
                break


            page_id = evidence_item[2]
            sentence_id = evidence_item[3]


            if (
                page_id is None
                or sentence_id is None
            ):

                set_is_complete = False
                break


            page_sentences = wiki_lookup.get(
                page_id
            )


            if page_sentences is None:

                set_is_complete = False
                break


            sentence_text = page_sentences.get(
                sentence_id
            )


            if sentence_text is None:

                set_is_complete = False
                break


            current_set.append(
                (
                    page_id,
                    sentence_id,
                    sentence_text
                )
            )


        if (
            set_is_complete
            and len(current_set) > 0
        ):

            valid_evidence_sets.append(
                current_set
            )


    if not valid_evidence_sets:

        missing_complete_evidence += 1
        continue


    unique_evidence = {}


    for evidence_set in valid_evidence_sets:

        for (
            page_id,
            sentence_id,
            sentence_text
        ) in evidence_set:

            key = (
                page_id,
                sentence_id
            )

            unique_evidence[key] = sentence_text


    sorted_evidence = sorted(
        unique_evidence.items(),
        key=lambda item: (
            item[0][0],
            item[0][1]
        )
    )


    evidence_texts = []
    evidence_pages = []
    evidence_sentence_ids = []


    for (
        page_id,
        sentence_id
    ), sentence_text in sorted_evidence:

        evidence_texts.append(
            sentence_text
        )

        evidence_pages.append(
            page_id
        )

        evidence_sentence_ids.append(
            sentence_id
        )


    evidence_text = " ".join(
        evidence_texts
    )


    output_rows.append(
        {
            "id": row["id"],
            "claim": row["claim"],
            "label": label,

            "evidence_text":
                evidence_text,

            "evidence_pages":
                json.dumps(
                    evidence_pages,
                    ensure_ascii=False
                ),

            "evidence_sentence_ids":
                json.dumps(
                    evidence_sentence_ids
                ),

            "num_evidence_sentences":
                len(evidence_texts),

            "num_evidence_sets":
                len(valid_evidence_sets)
        }
    )

    kept_rows += 1


df = pd.DataFrame(output_rows)


print(f"\nRows kept: {kept_rows:,}")

print(
    "Missing complete evidence:",
    f"{missing_complete_evidence:,}"
)

print("\nShape:")
print(df.shape)

print("\nLabels:")
print(
    df["label"].value_counts()
)


print("\nNumber of evidence sentences:")
print(
    df["num_evidence_sentences"]
    .value_counts()
    .sort_index()
)


print("\nNumber of complete evidence sets:")
print(
    df["num_evidence_sets"]
    .value_counts()
    .sort_index()
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)
