# measure retrieval precision and recall and save the results to an excel file
# for each alert, the correct records are the ones belonging to that patient

import json
import os
import pandas as pd

from chunking import build_documents
from retrieval import retrieve, alert_to_query

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERTS_PATH = os.path.join(BASE_DIR, "data", "rpm_alerts.json")
OUTPUT_XLSX = os.path.join(BASE_DIR, "retrieval_evaluation.xlsx")

K_VALUES = [1, 3, 5]


# how many chunks each patient has (their relevant set size)
def gold_counts():
    counts = {}
    for doc in build_documents():
        pid = doc["patient_id"]
        counts[pid] = counts.get(pid, 0) + 1
    return counts


def evaluate():
    with open(ALERTS_PATH, encoding="utf-8") as f:
        alerts = json.load(f)

    counts = gold_counts()
    rows = []

    for alert in alerts:
        pid = alert["patient_id"]
        total_relevant = counts.get(pid, 0)

        hits = retrieve(alert_to_query(alert), top_k=max(K_VALUES))
        found_pids = [h["patient_id"] for h in hits]

        # reciprocal rank = 1 / position of the first correct patient
        rr = 0.0
        for pos, fp in enumerate(found_pids, start=1):
            if fp == pid:
                rr = 1 / pos
                break

        row = {
            "alert_id": alert["alert_id"],
            "patient_id": pid,
            "top1": found_pids[0] if found_pids else None,
            "hit@1": 1 if found_pids and found_pids[0] == pid else 0,
            "reciprocal_rank": round(rr, 4),
        }
        for k in K_VALUES:
            correct = sum(1 for fp in found_pids[:k] if fp == pid)
            row["precision@" + str(k)] = round(correct / k, 4)
            row["recall@" + str(k)] = round(correct / total_relevant, 4) if total_relevant else 0
        rows.append(row)

    return pd.DataFrame(rows)


def make_summary(df):
    data = [["MRR", round(df["reciprocal_rank"].mean(), 4)],
            ["hit@1", round(df["hit@1"].mean(), 4)]]
    for k in K_VALUES:
        data.append(["precision@" + str(k), round(df["precision@" + str(k)].mean(), 4)])
        data.append(["recall@" + str(k), round(df["recall@" + str(k)].mean(), 4)])
    return pd.DataFrame(data, columns=["metric", "value"])


if __name__ == "__main__":
    df = evaluate()
    summary = make_summary(df)

    print(summary.to_string(index=False))
    print()
    print(df[["alert_id", "patient_id", "hit@1", "precision@3", "recall@3"]].to_string(index=False))

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        df.to_excel(writer, sheet_name="per_query", index=False)
    print("Saved retrieval_evaluation.xlsx")
