import json
from resume_summary import get_resume_data
from ranker import compute_score

def generate_html(ranked_resumes, output_file="ranked_resumes.html"):
    html_head = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Ranked Resumes</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .resume-box { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
            .resume-header { font-size: 1.5em; margin-bottom: 10px; color: #333; }
            .resume-score { font-size: 1.2em; color: #007BFF; }
            .resume-section { margin-bottom: 10px; }
            .section-title { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Ranked Resumes</h1>
    """
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
        <div class="resume-box">
            <div class="resume-header">{idx}. {name}</div>
            <div class="resume-score">Score: {score}</div>
            <div class="resume-section"><span class="section-title">Email:</span> {email}</div>
            <div class="resume-section"><span class="section-title">Phone:</span> {phone}</div>
            <div class="resume-section"><span class="section-title">Professional Summary:</span> {professional_summary}</div>
            <div class="resume-section"><span class="section-title">Skills:</span> {skills}</div>
            <div class="resume-section"><span class="section-title">Weaknesses:</span> {weaknesses}</div>
            <div class="resume-section"><span class="section-title">Areas for Improvement:</span> {improvements}</div>
        </div>
        """
        html_body += candidate_html
    html_tail = """
    </body>
    </html>
    """
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
