# AI-Powered Obstetric Emergency Decision Support System

A research-oriented, open-source prototype for obstetric emergency triage and decision support. This project combines clinical reasoning, transparent rule-based logic, and a modular software architecture to support the early assessment of high-risk obstetric conditions.

The current Version 1 implementation is a console-based application that evaluates patient-reported symptoms and classifies risk into Low Risk, Moderate Risk, High Risk, or Critical Risk. The architecture is intentionally designed so that a future machine learning model can replace the rule engine without changing the broader application workflow.

## Project Overview

Obstetric emergencies require rapid assessment, accurate triage, and timely intervention. This project aims to contribute to that objective by developing a decision support system that can assist clinicians and researchers in identifying potentially serious cases early.

The system is designed as a clinical decision-support tool, not a replacement for medical judgment. It should always be used in conjunction with professional medical assessment.

## Features

- Rule-based obstetric emergency assessment
- Console-based user interaction for Version 1
- Structured output including:
  - Risk Level
  - Risk Score
  - Clinical Explanation
  - Recommended Immediate Action
- Modular architecture for future expansion
- Reusable prediction module for integration into applications or services
- Transparent and explainable decision logic
- Clinical history checklist covering hypertension, diabetes, anemia, heart disease, multiple pregnancy, previous preeclampsia, and previous hemorrhage

See [CLINICAL_DECISION_RULES.md](CLINICAL_DECISION_RULES.md) for the current risk weights, classification levels, and combination rules.

## Project Architecture

The project follows a simple and modular structure:

- app.py: interactive console application for collecting patient information and displaying results
- predict.py: reusable prediction module containing the rule-based assessment engine
- train_model.py: planned training pipeline for future machine learning development
- data/: data storage and datasets
- models/: model artifacts and saved trained models
- templates/: web interface templates for future versions
- static/: static assets such as CSS and JavaScript
- notebooks/: experimentation and research notebooks
- screenshots/: visual documentation and project screenshots

## Folder Structure

```text
AI-Obstetric-Emergency-Decision-Support-System/
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── templates/
├── static/
├── notebooks/
└── screenshots/
```

## Installation

### Prerequisites

- Python 3.9 or higher

### Setup

```bash
git clone <repository-url>
cd AI-Obstetric-Emergency-Decision-Support-System
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### Run the application

```bash
run_app.bat
```

Or directly:

```bash
.venv\Scripts\python.exe -m flask --app app run --host 0.0.0.0 --port 5000
```

## Usage

When the program starts, it will prompt for the following clinical information:

- Patient age
- Pregnancy weeks
- Heavy bleeding
- Severe abdominal pain
- Blood pressure
- Body temperature
- Fetal movement
- Consciousness

The application then returns:

- Risk Level
- Risk Score
- Clinical Explanation
- Recommended Immediate Action

## Future Roadmap

### Version 1
- Console-based rule engine
- Transparent symptom-based risk classification
- Reusable prediction module

### Version 2
- Web-based interface using a lightweight framework
- Improved user experience and reporting

### Version 3
- Data-driven model integration
- Model training and validation pipeline

### Version 4
- Explainability and visualization enhancements
- Clinical decision support dashboards

### Version 5
- Deployment-ready health AI system with security, monitoring, and auditability

## Technologies

### Current Version
- Python 3
- Standard library only

### Planned Future Technologies
- Flask
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Research Background

This project is motivated by the need for timely, accurate, and explainable clinical decision support in obstetric care. Obstetric emergencies can progress rapidly, and early recognition of warning signs is essential for improving maternal and fetal outcomes.

The system is designed with a research perspective in mind, emphasizing transparency, modularity, and future extensibility for evidence-based medical AI development.

## License

This project is licensed under the MIT License.

## Author

Muhammad Zeeshan Javaid

---

This project is intended for academic, research, and educational use. It remains a prototype and should not be treated as a clinical diagnostic system.
