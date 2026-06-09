# ClinicalBridge

## Overview

ClinicalBridge is a proof-of-concept multi-agent clinical decision-support system developed for the COP-3442 Prompt Engineering Capstone Project.

The project aims to bridge the clinical context gap by integrating fragmented healthcare information from:

* Electronic Health Records (EHR)
* Remote Patient Monitoring (RPM)
* Anamnesis data (patient-reported information)

The system uses multiple LLM-powered agents to synthesize clinical context into structured summaries for healthcare professionals.

---

# Project Goals

The main objective of ClinicalBridge is to automatically combine fragmented healthcare data into actionable clinical context briefs.

The prototype demonstrates:

* prompt engineering
* multi-agent orchestration
* retrieval-augmented generation (RAG)
* hallucination evaluation
* healthcare safety guardrails

---

# Technologies Used

* Python
* LangChain
* Gemini / OpenAI API
* ChromaDB
* Pandas
* JSON

---

# Project Structure

```text
ClinicalBridge/
│
├── data/
├── evaluation/
├── reports/
├── prompts/
├── notebooks/
├── src/
```

---

# Dataset Components

The dataset includes:

* EHR patient records
* RPM device alerts
* anamnesis records
* clinical scenarios
* gold-standard outputs
* hallucination evaluation tables

---

# Team Responsibilities

## Student 1 — Dataset and Evaluation Lead

* simulated datasets
* clinical scenarios
* evaluation framework
* hallucination evaluation
* documentation

## Student 2 — Prompt Engineering Lead

* prompt design
* prompt iterations
* failure analysis
* safety guardrails

## Student 3 — RAG and Retrieval Lead

* vector database
* retrieval pipeline
* agent implementation

## Student 4 — Multi-Agent Integration Lead

* orchestration
* workflow management
* final prototype integration

---

# Educational Disclaimer

This project uses fully simulated patient data and is intended strictly for academic and educational purposes.
