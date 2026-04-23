"""
kg.py - Knowledge Graph Module
Loads symptom-disorder relations from JSON triples and queries them.
"""

import json
from typing import Dict, List, Tuple

# Mock Knowledge Graph: symptom-disorder relations
MOCK_KG_DATA = {
    "disorders": {
        "Depression": {
            "symptoms": ["sadness", "loss of interest", "fatigue", "sleep changes", 
                        "appetite changes", "guilt", "worthlessness", "concentration problems",
                        "hopelessness", "irritability"],
            "severity": "medium"
        },
        "Anxiety Disorder": {
            "symptoms": ["worry", "restlessness", "muscle tension", "sleep difficulties",
                        "irritability", "difficulty concentrating", "panic", "fear",
                        "shortness of breath", "heart racing"],
            "severity": "medium"
        },
        "Bipolar Disorder": {
            "symptoms": ["mood swings", "elevated mood", "high energy", "reduced sleep",
                        "racing thoughts", "impulsivity", "grandiosity", "depression",
                        "mania", "hypomania"],
            "severity": "high"
        },
        "Schizophrenia": {
            "symptoms": ["hallucinations", "delusions", "disorganized thinking",
                        "flat affect", "social withdrawal", "cognitive impairment",
                        "paranoia", "voices", "confusion"],
            "severity": "high"
        },
        "PTSD": {
            "symptoms": ["flashbacks", "nightmares", "avoidance", "hypervigilance",
                        "startle response", "emotional numbness", "intrusive thoughts",
                        "irritability", "guilt", "memory problems"],
            "severity": "high"
        },
        "OCD": {
            "symptoms": ["obsessions", "compulsions", "intrusive thoughts", "anxiety",
                        "checking", "cleaning", "ordering", "counting", "rituals"],
            "severity": "medium"
        },
        "ADHD": {
            "symptoms": ["inattention", "hyperactivity", "impulsivity", "distractibility",
                        "forgetfulness", "difficulty focusing", "fidgeting", "interrupting"],
            "severity": "low"
        },
        "Eating Disorder": {
            "symptoms": ["weight loss", "bingeing", "purging", "body image concerns",
                        "fear of weight gain", "food restriction", "vomiting", "excessive exercise"],
            "severity": "high"
        },
        "Panic Disorder": {
            "symptoms": ["panic attacks", "chest pain", "shortness of breath",
                        "dizziness", "choking", "numbness", "terror", "feeling losing control"],
            "severity": "medium"
        },
        "Insomnia": {
            "symptoms": ["difficulty sleeping", "early waking", "fatigue", "daytime sleepiness",
                        "poor concentration", "mood changes"],
            "severity": "low"
        }
    },
    "symptom_to_disorder": {
        "sadness": ["Depression"],
        "loss of interest": ["Depression"],
        "fatigue": ["Depression", "Insomnia"],
        "sleep changes": ["Depression", "Anxiety Disorder", "Insomnia"],
        "worry": ["Anxiety Disorder"],
        "restlessness": ["Anxiety Disorder", "ADHD"],
        "panic": ["Panic Disorder", "Anxiety Disorder"],
        "fear": ["Anxiety Disorder", "PTSD"],
        "flashbacks": ["PTSD"],
        "nightmares": ["PTSD", "Insomnia"],
        "hallucinations": ["Schizophrenia"],
        "delusions": ["Schizophrenia"],
        "mood swings": ["Bipolar Disorder"],
        "elevated mood": ["Bipolar Disorder"],
        "mania": ["Bipolar Disorder"],
        "obsessions": ["OCD"],
        "compulsions": ["OCD"],
        "inattention": ["ADHD"],
        "hyperactivity": ["ADHD"],
        "bingeing": ["Eating Disorder"],
        "purging": ["Eating Disorder"],
        "suicidal thoughts": ["Depression"],
        "self-harm": ["Depression", "Borderline Personality"],
        "hopelessness": ["Depression"],
        "irritability": ["Depression", "Anxiety Disorder", "PTSD"],
        "concentration problems": ["Depression", "Anxiety Disorder", "ADHD"],
        "intrusive thoughts": ["OCD", "PTSD"],
        "hypervigilance": ["PTSD"],
        "startle response": ["PTSD"]
    }
}


class KnowledgeGraph:
    """Knowledge Graph for symptom-disorder mapping."""
    
    def __init__(self, kg_data: Dict = None):
        """Initialize KG with data."""
        self.data = kg_data or MOCK_KG_DATA
        self.disorders = self.data["disorders"]
        self.symptom_map = self.data["symptom_to_disorder"]
    
    def get_disorders(self) -> List[str]:
        """Return all available disorders."""
        return list(self.disorders.keys())
    
    def get_symptoms_for_disorder(self, disorder: str) -> List[str]:
        """Get symptoms for a specific disorder."""
        return self.disorders.get(disorder, {}).get("symptoms", [])
    
    def get_disorders_for_symptom(self, symptom: str) -> List[str]:
        """Get disorders associated with a symptom."""
        return self.symptom_map.get(symptom.lower(), [])
    
    def compute_similarity(self, symptoms: List[str], disorder: str) -> float:
        """
        Compute overlap-based similarity between user symptoms and disorder symptoms.
        Returns a score between 0 and 1.
        """
        disorder_symptoms = set(self.get_symptoms_for_disorder(disorder))
        user_symptoms = set(s.lower() for s in symptoms)
        
        if not disorder_symptoms:
            return 0.0
        
        overlap = len(disorder_symptoms & user_symptoms)
        return overlap / len(disorder_symptoms)
    
    def query(self, symptoms: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Query KG with extracted symptoms.
        Returns top-k candidate disorders with similarity scores.
        """
        scores = {}
        for disorder in self.disorders:
            score = self.compute_similarity(symptoms, disorder)
            if score > 0:
                scores[disorder] = score
        
        # Sort by score descending
        sorted_disorders = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_disorders[:top_k]
    
    def get_severity(self, disorder: str) -> str:
        """Get severity level for a disorder."""
        return self.disorders.get(disorder, {}).get("severity", "unknown")


def load_kg_from_file(filepath: str) -> KnowledgeGraph:
    """Load KG from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return KnowledgeGraph(data)


if __name__ == "__main__":
    # Test the KG
    kg = KnowledgeGraph()
    test_symptoms = ["sadness", "loss of interest", "fatigue", "sleep changes"]
    results = kg.query(test_symptoms)
    print("KG Query Test:")
    print(f"Input symptoms: {test_symptoms}")
    print(f"Top matches: {results}")