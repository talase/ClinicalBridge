
from schemas import EHRQuery, RetrievedDocument, EHRResponse
from typing import List

class FakeRetrieval:
    def retrieve(self, query: str):
        return [
            {
                "doc_id": "doc1",
                "content": f"Medical note related to: {query}",
                "score": 0.87
            }
        ]

retrieval = FakeRetrieval()

class EHRAgent:

    def run(self, user_input: str) -> EHRResponse:

        query = EHRQuery(
            refined_query=user_input + " medical context",
            keywords=[]
        )

        docs_raw = retrieval.retrieve(query.refined_query)

        docs = [
            RetrievedDocument(**d) for d in docs_raw
        ]

        return EHRResponse(
            query=query,
            retrieved_context=docs,
            clinical_reasoning="Relevant clinical patterns detected.",
            final_answer="Further evaluation recommended."
        )
