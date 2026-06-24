
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.schemas import TriageOutput
import json

TRIAGE_SYSTEM = "You are a medical triage AI. Return JSON only."

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
    return TriageOutput(**chain.invoke({"patient_input": patient_input}))
