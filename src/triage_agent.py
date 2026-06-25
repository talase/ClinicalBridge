from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.schemas import TriageOutput
import json

TRIAGE_SYSTEM = """
You are a medical triage AI.

Return ONLY valid JSON.

Valid priority values:
- P1
- P2
- P3
- P4

Valid priority_label values:
- EMERGENT
- URGENT
- LESS_URGENT
- NON_URGENT

Mappings:
P1 -> EMERGENT
P2 -> URGENT
P3 -> LESS_URGENT
P4 -> NON_URGENT

Never use any other priority_label value.
Never output ROUTINE.
Never output text outside JSON.
"""

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

examples = [
    {
        "input": "chest pain radiating left arm sweating",
        "output": json.dumps({
            "priority": "P1",
            "priority_label": "EMERGENT",
            "pathway": "emergency_dispatch",
            "red_flags": ["chest_pain"],
            "escalate_human": True,
            "escalation_reason": "cardiac concern",
            "confidence": 0.95
        })
    },
    {
        "input": "mild headache for two days",
        "output": json.dumps({
            "priority": "P4",
            "priority_label": "NON_URGENT",
            "pathway": "primary_care_scheduling",
            "red_flags": [],
            "escalate_human": False,
            "escalation_reason": None,
            "confidence": 0.80
        })
    }
]

triage_prompt = ChatPromptTemplate.from_messages([
    ("system", TRIAGE_SYSTEM),
    FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    ),
    ("human", "{patient_input}")
])

parser = JsonOutputParser(pydantic_object=TriageOutput)


def build_triage_chain(llm):
    return triage_prompt | llm | parser


def run_triage(llm, patient_input: str):
    chain = build_triage_chain(llm)

    raw = chain.invoke({
        "patient_input": patient_input
    })

    print("\n===== RAW TRIAGE RESPONSE =====")
    print(json.dumps(raw, indent=2))
    print("================================\n")

    if raw.get("priority_label") == "ROUTINE":
        raw["priority_label"] = "NON_URGENT"

    return TriageOutput(**raw)
