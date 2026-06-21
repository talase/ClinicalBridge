# load the patient data, turn each record into text, and split into chunks

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EHR_PATH = os.path.join(BASE_DIR, "data", "ehr_records.json")
ANAMNESIS_PATH = os.path.join(BASE_DIR, "data", "anamnesis_records.json")

CHUNK_SIZE = 120     # words per chunk
CHUNK_OVERLAP = 20   # words shared between chunks


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_to_text(items):
    if not items:
        return "None"
    return ", ".join(str(x) for x in items)


def labs_to_text(labs):
    if not labs:
        return "None"
    return "; ".join(f"{k}: {v}" for k, v in labs.items())


def ehr_to_text(rec):
    return (
        f"EHR record\n"
        f"Patient ID: {rec['patient_id']}\n"
        f"Name: {rec['name']}, Age: {rec['age']}, Gender: {rec['gender']}\n"
        f"Diagnoses: {list_to_text(rec['diagnoses'])}\n"
        f"Medications: {list_to_text(rec['medications'])}\n"
        f"Allergies: {list_to_text(rec['allergies'])}\n"
        f"Recent labs: {labs_to_text(rec['recent_labs'])}"
    )


def anamnesis_to_text(rec):
    return (
        f"Anamnesis (patient reported)\n"
        f"Patient ID: {rec['patient_id']}\n"
        f"Symptoms: {list_to_text(rec['symptoms'])}\n"
        f"Medication adherence: {rec['medication_adherence']}\n"
        f"Lifestyle: {list_to_text(rec['lifestyle_notes'])}\n"
        f"Family history: {list_to_text(rec['family_history'])}"
    )


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


def build_documents():
    docs = []

    for rec in load_json(EHR_PATH):
        for i, chunk in enumerate(chunk_text(ehr_to_text(rec))):
            docs.append({
                "id": f"ehr_{rec['patient_id']}_{i}",
                "text": chunk,
                "patient_id": rec["patient_id"],
                "source": "ehr",
            })

    for rec in load_json(ANAMNESIS_PATH):
        for i, chunk in enumerate(chunk_text(anamnesis_to_text(rec))):
            docs.append({
                "id": f"anamnesis_{rec['patient_id']}_{i}",
                "text": chunk,
                "patient_id": rec["patient_id"],
                "source": "anamnesis",
            })

    return docs


if __name__ == "__main__":
    ehr = load_json(EHR_PATH)
    anam = load_json(ANAMNESIS_PATH)
    print("EHR records:", len(ehr))
    print("Anamnesis records:", len(anam))
    print()
    print(ehr_to_text(ehr[0]))
    print()
    print("Total chunks:", len(build_documents()))
