import pandas as pd


INPUT_FILE = "data/FEVER/fever_experiment_v2.csv"
OUTPUT_FILE = "data/FEVER/fever_framed_v2.csv"

templates = {

    "positive": [

        (
            "This information may inspire enthusiasm and positive feelings. "
            "Consider the following statement carefully."
        ),

        (
            "This information may be received with interest and optimism. "
            "Consider the following statement carefully."
        ),

        (
            "This information may evoke a sense of appreciation and enthusiasm. "
            "Consider the following statement carefully."
        ),

        (
            "The information may leave a particularly encouraging impression. "
            "Consider the following statement carefully."
        ),

        (
            "The information may leave the reader feeling encouraged and reassured. "
            "Consider the following statement carefully."
        )
    ],


    "negative": [

        (
            "This information may evoke concern and unease. "
            "Consider the following statement carefully."
        ),

        (
            "This information may be received with hesitation and discomfort. "
            "Consider the following statement carefully."
        ),

        (
            "This information may leave a somewhat troubling impression. "
            "Consider the following statement carefully."
        ),

        (
            "The information may raise doubts and create an uneasy impression. "
            "Consider the following statement carefully."
        ),

        (
            "The information may leave the reader feeling unsettled and concerned. "
            "Consider the following statement carefully."
        )
    ],


    "fear": [

        (
            "This information may be received with caution and unease. "
            "Consider the following statement carefully."
        ),

        (
            "This information may leave a somewhat unsettling impression. "
            "Consider the following statement carefully."
        ),

        (
            "The information may leave the reader feeling uneasy and uncertain. "
            "Consider the following statement carefully."
        ),

        (
            "This information may evoke a sense of vulnerability and uneasiness. "
            "Consider the following statement carefully."
        ),

        (
            "This information may be met with apprehension and caution. "
            "Consider the following statement carefully."
        )
    ],


    "anger": [

        (
            "This information may evoke frustration and irritation. "
            "Consider the following statement carefully."
        ),

        (
            "This information may be received with displeasure and frustration. "
            "Consider the following statement carefully."
        ),

        (
            "This information may leave a somewhat frustrating impression. "
            "Consider the following statement carefully."
        ),

        (
            "The information may be met with frustration and disapproval. "
            "Consider the following statement carefully."
        ),

        (
            "The information may leave the reader feeling irritated and dissatisfied. "
            "Consider the following statement carefully."
        )
    ]
}


df = pd.read_csv(INPUT_FILE)

print("Shape:", df.shape)

print("\nLabels:")
print(df["label"].value_counts())


results = []

emotions = [
    "positive",
    "negative",
    "fear",
    "anger"
]



for label in ["SUPPORTS", "REFUTES"]:

    label_df = df[
        df["label"] == label
    ].copy()


    label_df = label_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)


    for row_index, row in label_df.iterrows():

        results.append({

            "id": row["id"],

            "original_claim": row["claim"],

            "label": row["label"],

            "evidence_text": row["evidence_text"],

            "num_evidence_sentences":
                row["num_evidence_sentences"],

            "emotion": "neutral",

            "template_id": "neutral",

            "text": row["claim"]
        })


        for emotion in emotions:

            template_index = row_index % 5

            template = templates[
                emotion
            ][template_index]

            template_id = (
                f"{emotion}_{template_index + 1}"
            )


            framed_text = (
                template
                + " "
                + row["claim"]
            )


            results.append({

                "id": row["id"],

                "original_claim": row["claim"],

                "label": row["label"],

                "evidence_text":
                    row["evidence_text"],

                "num_evidence_sentences":
                    row[
                        "num_evidence_sentences"
                    ],

                "emotion": emotion,

                "template_id": template_id,

                "text": framed_text
            })


framed_df = pd.DataFrame(results)

emotion_order = {

    "neutral": 0,
    "positive": 1,
    "negative": 2,
    "fear": 3,
    "anger": 4
}


framed_df["emotion_order"] = (
    framed_df["emotion"]
    .map(emotion_order)
)


framed_df = framed_df.sort_values(
    by=[
        "id",
        "emotion_order"
    ]
).reset_index(drop=True)


framed_df = framed_df.drop(
    columns=["emotion_order"]
)



framed_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Shape:", framed_df.shape)

print("\nEmotion distribution:")
print(
    framed_df["emotion"]
    .value_counts()
)

print("\nLabel distribution:")
print(
    framed_df["label"]
    .value_counts()
)

print("\nTemplate distribution:")
print(
    framed_df["template_id"]
    .value_counts()
    .sort_index()
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)
