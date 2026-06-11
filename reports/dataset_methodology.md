# Dataset Generation Methodology

## Overview

The ClinicalBridge project uses a simulated healthcare dataset designed to mimic fragmented clinical information from multiple healthcare sources. The dataset was created for educational and research purposes only and does not contain any real patient information.

The dataset integrates three major healthcare data categories:

* Electronic Health Records (EHR)
* Remote Patient Monitoring (RPM) alerts
* Anamnesis records (patient-reported information)

The purpose of the dataset is to evaluate how a multi-agent LLM system can synthesize fragmented clinical information into a unified clinical context brief.

---

# Dataset Structure

The dataset was divided into four primary JSON files:

## 1. EHR Records

Contains:

* patient demographics
* diagnoses
* medications
* allergies
* laboratory results

This file simulates traditional electronic medical records stored in hospitals.

---

## 2. RPM Alerts

Contains:

* device alerts
* physiological readings
* urgency levels
* timestamps

This file simulates data generated from wearable devices and remote monitoring systems.

---

## 3. Anamnesis Records

Contains:

* symptoms
* medication adherence
* lifestyle information
* family history

This file simulates patient-reported information typically collected during clinical interviews.

---

## 4. Clinical Scenarios

Contains:

* scenario descriptions
* linked alerts
* expected system behavior

These scenarios were designed to test the ability of the ClinicalBridge system to synthesize clinical context across fragmented data sources.

---

# Dataset Design Principles

The dataset was intentionally designed with realistic imperfections to simulate real-world healthcare environments.

Examples include:

* incomplete patient records
* missing laboratory data
* conflicting information
* sparse patient histories
* varied alert severity levels

These imperfections allow the evaluation framework to test system robustness and hallucination resistance.

---

# Clinical Diversity

The dataset includes multiple clinical conditions, including:

* hypertension
* diabetes
* heart failure
* asthma
* chronic kidney disease
* COPD
* sleep apnea
* obesity

This diversity ensures broader evaluation coverage across different alert and symptom types.

---

# Educational Purpose

The dataset was developed exclusively for academic demonstration purposes within the Prompt Engineering capstone project. No real patient data was used at any stage of development.
