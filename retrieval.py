# search functions used by the agents to pull patient context from the store
"""
from embedding import COLLECTION_NAME, embed_texts, get_client

TOP_K = 5


def retrieve(query, top_k=TOP_K, patient_id=None, source=None):
    collection = get_client().get_collection(COLLECTION_NAME)

    if patient_id and source:
        where = {"$and": [{"patient_id": patient_id}, {"source": source}]}
    elif patient_id:
        where = {"patient_id": patient_id}
    elif source:
        where = {"source": source}
    else:
        where = None

    query_vec = embed_texts([query])
    res = collection.query(query_embeddings=query_vec, n_results=top_k, where=where)

    hits = []
    for i in range(len(res["ids"][0])):
        dist = float(res["distances"][0][i])
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "patient_id": res["metadatas"][0][i]["patient_id"],
            "source": res["metadatas"][0][i]["source"],
            "distance": round(dist, 4),
            "score": round(1 - dist, 4),
        })
    return hits


def retrieve_patient_context(patient_id, query, top_k=TOP_K, source=None):
    # an alert is always about one patient, so we filter to that patient
    return retrieve(query, top_k=top_k, patient_id=patient_id, source=source)


def alert_to_query(alert):
    return f"{alert['alert_type']} {alert['reading']} from {alert['device_type']}"


if __name__ == "__main__":
    import json

    print("Open search: high blood pressure, stopped medication")
    for h in retrieve("high blood pressure spike, stopped taking medication", top_k=3):
        print(" ", h["score"], h["source"], h["patient_id"])

    with open("data/rpm_alerts.json", encoding="utf-8") as f:
        alerts = json.load(f)

    alert = alerts[0]
    print()
    print("Alert", alert["alert_id"], "for patient", alert["patient_id"])
    for h in retrieve_patient_context(alert["patient_id"], alert_to_query(alert), top_k=4):
        print(" ", h["score"], h["source"])
"""


def retrieve(query, top_k=5, patient_id=None, source=None):

    try:
        collection = get_client().get_collection(COLLECTION_NAME)
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        return results.get("documents", [[]])[0]

    except Exception:
        # HARD FALLBACK (prevents crash)
        return [
            {
                "id": "fallback",
                "content": "No retrieval database available. Using safe fallback context.",
                "score": 0.0
            }
        ]
