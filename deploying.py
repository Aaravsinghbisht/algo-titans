import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Optional

MODEL_DIR = "./AreaRecommender_model"

class ResumeRecommender:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        if torch.cuda.is_available():
            self.model.to("cuda")

    def generate_recommendation(self, category: str, resume_quality: str, weakness: str, 
                              max_length: int = 256, num_beams: int = 5, 
                              do_sample: bool = True, temperature: float = 0.7, top_k: int = 50, 
                              top_p: float = 0.95) -> Dict[str, str]:
        prompt = f"Category: {category}\nResume Quality: {resume_quality}\nWeakness: {weakness}\nRecommendation:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            early_stopping=True
        )
        
        recommendation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {
            "category": category,
            "resume_quality": resume_quality,
            "weakness": weakness,
            "recommendation": recommendation
        }

    def analyze_resume_section(self, section_data: Dict[str, str]) -> Dict[str, str]:
        return self.generate_recommendation(
            category=section_data.get("category", ""),
            resume_quality=section_data.get("resume_quality", ""),
            weakness=section_data.get("weakness", "")
        )

def get_recommendations(resume_data: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    recommender = ResumeRecommender()
    recommendations = {}
    
    for section, data in resume_data.items():
        if isinstance(data, dict) and all(key in data for key in ["category", "resume_quality", "weakness"]):
            recommendations[section] = recommender.analyze_resume_section(data)
    
    return recommendations

if __name__ == "__main__":
    # Example usage
    sample_resume_data = {
        "technical_skills": {
            "category": "Technical Skills",
            "resume_quality": "Fair",
            "weakness": "Limited experience with advanced Python libraries"
        },
        "education": {
            "category": "Education",
            "resume_quality": "Good",
            "weakness": "No relevant certifications"
        }
    }
    
    recommendations = get_recommendations(sample_resume_data)
    print(recommendations)