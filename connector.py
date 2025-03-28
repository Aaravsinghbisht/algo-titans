import json
from resume_summary import get_resume_data
from ranker import compute_score

html_head = """
<!DOCTYPE html>
<html>
<head>
    <title>Ranked Resumes</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .candidate { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
        .score { color: #2196F3; font-weight: bold; }
    </style>
</head>
<body>
"""

html_tail = """
</body>
</html>
"""

def generate_html(ranked_resumes, output_file="ranked_resumes.html"):
    
    html_body = ""
    for idx, candidate in enumerate(ranked_resumes, start=1):
        name = candidate.get("Name") or candidate.get("name", "Unknown")
        score = candidate.get("score", 0)
        if isinstance(candidate.get("Contact Information"), dict):
            email = candidate.get("Contact Information", {}).get("Email", "N/A")
            phone = candidate.get("Contact Information", {}).get("Phone", "N/A")
        else:
            email = candidate.get("Email", "N/A")
            phone = candidate.get("Phone", "N/A")
        professional_summary = candidate.get("Professional Summary", "Not available")
        skills = candidate.get("Skills", "Not available")
        weaknesses = candidate.get("weaknesses", "Not available")
        improvements = candidate.get("Areas for Improvement", candidate.get("improvements", "Not available"))
        
        candidate_html = f"""
        <div class="candidate">
            <h2>{idx}. {name} <span class="score">Score: {score}</span></h2>
            <p><strong>Contact:</strong> {email} | {phone}</p>
            <p><strong>Summary:</strong> {professional_summary}</p>
            <p><strong>Skills:</strong> {skills}</p>
            <p><strong>Weaknesses:</strong> {weaknesses}</p>
            <p><strong>Improvements:</strong> {improvements}</p>
        </div>
        """
        html_body += candidate_html
   
    html_content = html_head + html_body + html_tail
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML page generated: {output_file}")

def main():
    resumes = get_resume_data()
    for candidate in resumes:
        candidate["score"] = compute_score(candidate)
    ranked_resumes = sorted(resumes, key=lambda x: x["score"], reverse=True)
    generate_html(ranked_resumes)

if __name__ == "__main__":
    main()

