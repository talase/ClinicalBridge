import os
import sys
import json
import csv
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from orchestrator import ClinicalOrchestrator


# ==========================================================
# CONFIGURATION
# ==========================================================

GOLD_STANDARD = "evaluation/gold_standard_outputs.json"
OUTPUT_DIR = "evaluation/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def keyword_score(expected_summary: str, generated_summary: str):
    """
    Simple keyword-overlap score.
    """

    stop_words = {
        "the", "a", "an", "and", "or", "to",
        "of", "for", "with", "may", "likely",
        "patient", "system", "should"
    }

    expected = {
        w.lower().strip(".,")
        for w in expected_summary.split()
        if len(w) > 3
    }

    expected = expected - stop_words

    generated = generated_summary.lower()

    hits = 0

    for word in expected:
        if word in generated:
            hits += 1

    if len(expected) == 0:
        return 1.0

    return round(hits / len(expected), 2)


def contains_hallucination(text):

    forbidden = [
        "cancer",
        "pregnancy",
        "stroke confirmed",
        "heart attack confirmed",
        "covid",
        "chemotherapy",
        "dialysis",
        "brain tumor"
    ]

    text = text.lower()

    for word in forbidden:
        if word in text:
            return True

    return False


# ==========================================================
# LOAD GOLD STANDARD
# ==========================================================

with open(GOLD_STANDARD, "r", encoding="utf8") as f:
    scenarios = json.load(f)

print("=" * 70)
print("ClinicalBridge Evaluation")
print("=" * 70)

print(f"\nLoaded {len(scenarios)} scenarios.\n")


# ==========================================================
# CREATE MODEL
# ==========================================================

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

orch = ClinicalOrchestrator(llm)


# ==========================================================
# STORAGE
# ==========================================================

rows = []

priority_correct = 0
risk_correct = 0
hallucination_count = 0
keyword_scores = []


# ==========================================================
# RUN EVERY SCENARIO
# ==========================================================

for scenario in scenarios:

    print("=" * 60)
    print(scenario["scenario_id"])
    print("=" * 60)

    print(scenario["input"])
    print()

    result = orch.run_all(scenario["input"])

    triage = result["triage"]
    synthesis = result["synthesis"]

    actual_priority = triage["priority"]
    actual_risk = synthesis["overall_risk"]
    narrative = synthesis["clinical_narrative"]

    expected_priority = scenario["expected_priority"]
    expected_risk = scenario["expected_risk"]
    expected_summary = scenario["expected_summary"]

    priority_match = actual_priority == expected_priority
    risk_match = actual_risk == expected_risk

    if priority_match:
        priority_correct += 1

    if risk_match:
        risk_correct += 1

    score = keyword_score(
        expected_summary,
        narrative
    )

    keyword_scores.append(score)

    hallucination = contains_hallucination(
        narrative
    )

    if hallucination:
        hallucination_count += 1

    print(f"Priority : {expected_priority} -> {actual_priority}")
    print(f"Risk     : {expected_risk} -> {actual_risk}")
    print(f"Keywords : {score:.2f}")
    print(f"Hallucination : {hallucination}")

    rows.append({

        "Scenario": scenario["scenario_id"],

        "Expected Priority": expected_priority,
        "Actual Priority": actual_priority,
        "Priority Match": priority_match,

        "Expected Risk": expected_risk,
        "Actual Risk": actual_risk,
        "Risk Match": risk_match,

        "Keyword Score": score,

        "Hallucination": hallucination

    })






# ==========================================================
# CREATE DATAFRAME
# ==========================================================

df = pd.DataFrame(rows)

comparison_csv = Path(OUTPUT_DIR) / "comparison_table.csv"
df.to_csv(comparison_csv, index=False)

print("\nSaved:", comparison_csv)


# ==========================================================
# METRICS
# ==========================================================

total_cases = len(df)

priority_accuracy = round(priority_correct / total_cases * 100, 2)
risk_accuracy = round(risk_correct / total_cases * 100, 2)
average_keyword_score = round(sum(keyword_scores) / total_cases * 100, 2)
hallucination_rate = round(hallucination_count / total_cases * 100, 2)

metrics = pd.DataFrame([
    {
        "Metric": "Priority Accuracy",
        "Value": priority_accuracy
    },
    {
        "Metric": "Risk Accuracy",
        "Value": risk_accuracy
    },
    {
        "Metric": "Average Keyword Score",
        "Value": average_keyword_score
    },
    {
        "Metric": "Hallucination Rate",
        "Value": hallucination_rate
    }
])

metrics_csv = Path(OUTPUT_DIR) / "evaluation_metrics.csv"
metrics.to_csv(metrics_csv, index=False)

print("Saved:", metrics_csv)


# ==========================================================
# HALLUCINATION REPORT
# ==========================================================

hallucination_df = df[
    ["Scenario", "Hallucination"]
]

hall_csv = Path(OUTPUT_DIR) / "hallucination_results.csv"
hallucination_df.to_csv(hall_csv, index=False)

print("Saved:", hall_csv)


# ==========================================================
# CHARTS
# ==========================================================

figures = Path(OUTPUT_DIR) / "figures"
figures.mkdir(exist_ok=True)


def save_bar_chart(title, labels, values, filename):

    plt.figure(figsize=(6, 4))

    plt.bar(labels, values)

    plt.ylim(0, 100)

    plt.title(title)

    plt.ylabel("Percentage")

    for i, value in enumerate(values):
        plt.text(
            i,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(figures / filename, dpi=300)

    plt.close()


save_bar_chart(
    "Priority Accuracy",
    ["Priority"],
    [priority_accuracy],
    "priority_accuracy.png"
)

save_bar_chart(
    "Risk Accuracy",
    ["Risk"],
    [risk_accuracy],
    "risk_accuracy.png"
)

save_bar_chart(
    "Keyword Similarity",
    ["Similarity"],
    [average_keyword_score],
    "keyword_similarity.png"
)

save_bar_chart(
    "Hallucination Rate",
    ["Hallucinations"],
    [hallucination_rate],
    "hallucination_rate.png"
)


# ==========================================================
# OVERALL SUMMARY CHART
# ==========================================================

plt.figure(figsize=(8, 5))

labels = [
    "Priority",
    "Risk",
    "Keywords",
    "Hallucinations"
]

values = [
    priority_accuracy,
    risk_accuracy,
    average_keyword_score,
    100 - hallucination_rate
]

plt.bar(labels, values)

plt.ylim(0, 100)

plt.ylabel("Score (%)")

plt.title("ClinicalBridge Overall Evaluation")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 1,
        f"{value:.1f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    figures / "overall_results.png",
    dpi=300
)

plt.close()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Priority Accuracy     : {priority_accuracy:.2f}%")
print(f"Risk Accuracy         : {risk_accuracy:.2f}%")
print(f"Keyword Similarity    : {average_keyword_score:.2f}%")
print(f"Hallucination Rate    : {hallucination_rate:.2f}%")

print("\nEvaluation completed successfully.")

print(f"\nResults saved to:\n{OUTPUT_DIR}")

print("\nGenerated files:")

print("  comparison_table.csv")
print("  evaluation_metrics.csv")
print("  hallucination_results.csv")

print("\nFigures:")

print("  priority_accuracy.png")
print("  risk_accuracy.png")
print("  keyword_similarity.png")
print("  hallucination_rate.png")
print("  overall_results.png")
