import re
import argparse

from numpy import extract

def extract_contact_info(text):

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = lines[0] if lines else None

    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone_match = re.search(r'(\+?\d[\d \-\(\)]{8,}\d)', text)
    
    return {
        "Name": name,
        "Email": email_match.group() if email_match else None,
        "Phone": phone_match.group() if phone_match else None
    }

def extract_section(text, section_titles):
   
    pattern = r'(?i)(' + '|'.join(re.escape(title) for title in section_titles) + r')'
    splits = re.split(pattern, text)
    
    sections = {}
    if len(splits) >= 3:
        
        for i in range(1, len(splits)-1, 2):
            header = splits[i].strip()
            content = splits[i+1].strip()
            sections[header] = content
    return sections

def extract_professional_summary(text):
    sections = extract_section(text, ["Professional Summary", "Summary", "Objective"])
    return next(iter(sections.values()), None)

def extract_work_experience(text):
    sections = extract_section(text, ["Work Experience", "Professional Experience", "Experience"])
    return next(iter(sections.values()), None)

def extract_education(text):
    sections = extract_section(text, ["Education", "Academic Background"])
    return next(iter(sections.values()), None)

def extract_skills(text):
    sections = extract_section(text, ["Skills", "Technical Skills", "Key Skills"])
    return next(iter(sections.values()), None)

def extract_projects(text):
    sections = extract_section(text, ["Projects", "Project Experience"])
    return next(iter(sections.values()), None)

def analyze_resume(text):

    analysis = {}
    analysis["Contact Information"] = extract_contact_info(text)
    analysis["Professional Summary"] = extract_professional_summary(text)
    analysis["Work Experience"] = extract_work_experience(text)
    analysis["Education"] = extract_education(text)
    analysis["Skills"] = extract_skills(text)
    analysis["Projects"] = extract_projects(text)
    analysis["Areas for Improvement"] = "Not analyzed automatically (manual review recommended)"
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="AI-based Resume Analyzer")
    parser.add_argument("filepath", help="Path to the resume text file")
    args = parser.parse_args()
    
    try:
        with open(args.filepath, "r", encoding="utf-8") as file:
            resume_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    analysis = analyze_resume(resume_text)
    
    print("====== Resume Analysis Summary ======\n")
    for section, content in analysis.items():
        print(f"--- {section} ---")
        print(content if content else "Not found")
        print()

if __name__ == "__main__":
    main()
