from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLOR_EVIDENCE = "#062D6D"
COLOR_NO_EVIDENCE = "#E58A2B"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


RESULTS_DIR = Path("results")

fcr = pd.read_csv(
    RESULTS_DIR / "h1_fcr_confidence_intervals.csv"
)



emotion_order = [
    "positive",
    "negative",
    "fear",
    "anger",
]

present = (
    fcr[
        fcr["evidence_condition"] == "present"
    ]
    .set_index("emotion")
    .loc[emotion_order]
)

absent = (
    fcr[
        fcr["evidence_condition"] == "absent"
    ]
    .set_index("emotion")
    .loc[emotion_order]
)

x = list(range(len(emotion_order)))

offset = 0.10

x_present = [
    value - offset
    for value in x
]

x_absent = [
    value + offset
    for value in x
]


present_fcr = (
    present["fcr"] * 100
)

absent_fcr = (
    absent["fcr"] * 100
)


present_yerr = [
    (
        present["fcr"]
        -
        present["ci_lower"]
    ) * 100,
    (
        present["ci_upper"]
        -
        present["fcr"]
    ) * 100,
]

absent_yerr = [
    (
        absent["fcr"]
        -
        absent["ci_lower"]
    ) * 100,
    (
        absent["ci_upper"]
        -
        absent["fcr"]
    ) * 100,
]



fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.errorbar(
    x_present,
    present_fcr,
    yerr=present_yerr,
    fmt="o",
    color=COLOR_EVIDENCE,
    markersize=8,
    capsize=4,
    elinewidth=1.8,
    capthick=1.8,
    label="Evidence present",
)

ax.errorbar(
    x_absent,
    absent_fcr,
    yerr=absent_yerr,
    fmt="o",
    color=COLOR_NO_EVIDENCE,
    markersize=8,
    capsize=4,
    elinewidth=1.8,
    capthick=1.8,
    label="Evidence absent",
)


ax.set_xticks(x)

ax.set_xticklabels(
    [
        "Positive",
        "Negative",
        "Fear",
        "Anger",
    ]
)

ax.set_ylabel(
    "Framing Change Rate (%)"
)

ax.set_xlabel(
    "Emotional framing"
)

ax.set_title(
    "Emotional Framing Rarely Changed Gemini's Predictions",
    pad=16,
)

ax.set_ylim(
    0,
    1.30,
)

ax.legend(
    loc="upper center",
    ncol=2,
    frameon=False,
)

ax.set_axisbelow(True)

ax.grid(
    axis="y",
    alpha=0.18,
    linewidth=0.8,
)



plt.tight_layout()

png_path = (
    RESULTS_DIR
    / "figure_1_fcr_by_emotion.png"
)

pdf_path = (
    RESULTS_DIR
    / "figure_1_fcr_by_emotion.pdf"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)


plt.show()


accuracy = pd.read_csv(
    RESULTS_DIR / "accuracy_inference.csv"
)

framing_order = [
    "neutral",
    "positive",
    "negative",
    "fear",
    "anger",
]

accuracy = (
    accuracy
    .set_index("emotion")
    .loc[framing_order]
)

x = list(range(len(framing_order)))

offset = 0.10

x_present = [
    value - offset
    for value in x
]

x_absent = [
    value + offset
    for value in x
]


accuracy_present = (
    accuracy["accuracy_present"] * 100
)

accuracy_absent = (
    accuracy["accuracy_absent"] * 100
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.scatter(
    x_present,
    accuracy_present,
    color=COLOR_EVIDENCE,
    s=70,
    label="Evidence present",
    zorder=3,
)

ax.scatter(
    x_absent,
    accuracy_absent,
    color=COLOR_NO_EVIDENCE,
    s=70,
    label="Evidence absent",
    zorder=3,
)


for i in range(len(x)):

    ax.plot(
        [
            x_present[i],
            x_absent[i],
        ],
        [
            accuracy_present.iloc[i],
            accuracy_absent.iloc[i],
        ],
        color="#B8B8B8",
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )

ax.set_xticks(x)

ax.set_xticklabels(
    [
        "Neutral",
        "Positive",
        "Negative",
        "Fear",
        "Anger",
    ]
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_xlabel(
    "Framing condition"
)

ax.set_title(
    "Explicit Evidence Consistently Improved Factual Accuracy",
    pad=16,
)

ax.legend(
    loc="upper center",
    ncol=2,
    frameon=False,
)

ax.set_axisbelow(True)

ax.grid(
    axis="y",
    alpha=0.18,
    linewidth=0.8,
)


ax.set_ylim(
    93.5,
    97.5,
)


plt.tight_layout()

png_path = (
    RESULTS_DIR
    / "figure_2_accuracy.png"
)

pdf_path = (
    RESULTS_DIR
    / "figure_2_accuracy.pdf"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)


plt.show()



h2_effect = 0.117
h2_ci_lower = -0.334
h2_ci_upper = 0.567


h2_xerr = [
    [h2_effect - h2_ci_lower],
    [h2_ci_upper - h2_effect],
]


fig, ax = plt.subplots(
    figsize=(8, 3.6)
)


ax.axvline(
    x=0,
    color="#777777",
    linewidth=1.5,
    linestyle="--",
    zorder=1,
)


ax.errorbar(
    h2_effect,
    0,
    xerr=h2_xerr,
    fmt="o",
    color=COLOR_EVIDENCE,
    markersize=9,
    capsize=5,
    elinewidth=2,
    capthick=2,
    zorder=3,
)


ax.set_title(
    "No Significant Reduction in Framing Sensitivity With Evidence",
    pad=16,
)

ax.set_xlabel(
    "FCR difference: No evidence - Evidence (percentage points)"
)

ax.set_yticks([])

ax.set_xlim(
    -0.6,
    0.8,
)

ax.set_ylim(
    -0.6,
    0.6,
)

ax.set_axisbelow(True)

ax.grid(
    axis="x",
    alpha=0.18,
    linewidth=0.8,
)


ax.text(
    h2_effect,
    0.20,
    "+0.117 pp",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="semibold",
)

ax.text(
    h2_effect,
    -0.20,
    "95% CI [-0.334, 0.567]\np = .661",
    ha="center",
    va="top",
    fontsize=11,
)


plt.tight_layout()

png_path = (
    RESULTS_DIR
    / "figure_3_primary_h2_effect.png"
)

pdf_path = (
    RESULTS_DIR
    / "figure_3_primary_h2_effect.pdf"
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()
