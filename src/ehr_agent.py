import json
import os

from src.schemas import (
    EHRQuery,
    RetrievedDocument,
    EHRResponse
)


class EHRAgent:

    def __init__(self):

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        ehr_path = os.path.join(
            root,
            "data",
            "ehr_records.json"
        )

        with open(ehr_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def retrieve(self, query: str):

        query = query.lower()

        results = []

        for patient in self.records:

            searchable_text = json.dumps(patient).lower()

            score = 0

            for word in query.split():

                if word in searchable_text:
                    score += 1

            if score > 0:

                results.append(
                    {
                        "doc_id": patient["patient_id"],
                        "content": json.dumps(patient),
                        "score": score
                    }
                )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:3]

    def run(self, user_input: str) -> EHRResponse:

        query = EHRQuery(
            refined_query=user_input,
            keywords=[]
        )

        docs_raw = self.retrieve(
            query.refined_query
        )

        docs = [
            RetrievedDocument(**d)
            for d in docs_raw
        ]

        if docs:

            reasoning = (
                "Relevant patient records retrieved "
                "from simulated EHR database."
            )

            answer = (
                "Patient history available "
                "for physician review."
            )

        else:

            reasoning = (
                "No matching EHR records found."
            )

            answer = (
                "Manual review recommended."
            )

        return EHRResponse(
            query=query,
            retrieved_context=docs,
            clinical_reasoning=reasoning,
            final_answer=answer
        )
