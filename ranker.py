import json
from resume_summary import get_resume_data  # This module should provide a function returning resume data as a list of dictionaries.

def compute_score(candidate):
    return len(candidate.get("skills", [])) - len(candidate.get("improvements", [])) - len(candidate.get("weaknesses", []))

def main():
    resumes = get_resume_data()
    for candidate in resumes:
        candidate["score"] = compute_score(candidate)
    ranked_resumes = sorted(resumes, key=lambda x: x["score"], reverse=True)
    print("Ranked Candidates:")
    for idx, candidate in enumerate(ranked_resumes, start=1):
        print(f"{idx}. {candidate['name']} - Score: {candidate['score']}")
    with open("ranked_resumes.json", "w", encoding="utf-8") as f:
        json.dump(ranked_resumes, f, indent=4)

if __name__ == "__main__":
    main()
