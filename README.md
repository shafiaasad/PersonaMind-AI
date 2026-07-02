# PersonaMind AI 2.0 - Final High Accuracy Version

This project is separated into frontend and backend files.

## Main Files

- `frontend.py` = Streamlit frontend / user interface
- `backend.py` = AI backend / dataset / ML models / predictions
- `data/sample_persona_dataset.csv` = dataset CSV
- `data/sample_persona_dataset.xlsx` = dataset Excel file
- `assets/logo.png` = project logo

## AI Models Used

- KNN for career prediction
- Neural Network / MLPClassifier for career prediction
- K-Means for behavior clustering
- NLP TF-IDF for text-based career prediction
- Rule-based AI advisor for personality, learning style, roadmap, and advice

## Accuracy Improvement

The dataset has been improved with clearer career-wise patterns and more records. This increases the demo model accuracy compared with the previous version.

## Run Project

```bash
pip install -r requirements.txt
streamlit run frontend.py
```

## Folder Structure

```text
PersonaMind_AI_2_0_HIGH_ACCURACY_FINAL/
|-- frontend.py
|-- backend.py
|-- README.md
|-- requirements.txt
|-- START_HERE.txt
|-- assets/
|   |-- logo.png
|   |-- persona-hero.svg
|   `-- profile-lab.svg
`-- data/
    |-- sample_persona_dataset.csv
    |-- sample_persona_dataset.xlsx
    `-- README.md
```

