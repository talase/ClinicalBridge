from src.schemas import TriageOutput
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class TriageOutput(BaseModel):
    priority: Literal["P1", "P2", "P3", "P4"]
    priority_label: Literal["EMERGENT", "URGENT", "LESS_URGENT", "NON_URGENT"]
    pathway: str
    red_flags: List[str]
    escalate_human: bool
    escalation_reason: Optional[str]
    confidence: float


class EHRQuery(BaseModel):
    refined_query: str
    keywords: List[str]


class RetrievedDocument(BaseModel):
    doc_id: str
    content: str
    score: Optional[float] = None


class EHRResponse(BaseModel):
    query: EHRQuery
    retrieved_context: List[RetrievedDocument]
    clinical_reasoning: str
    final_answer: str


class SymptomItem(BaseModel):
    symptom: str
    duration: Optional[str] = None
    severity: Optional[str] = None


class AnamnesisOutput(BaseModel):
    symptoms: List[SymptomItem]
    adherence_issues: List[str]
    risk_flags: List[str]
    summary: str
