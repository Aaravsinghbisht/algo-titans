import json

def get_resume_data():
    try:
        with open("resume_analysis_results.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return [] 