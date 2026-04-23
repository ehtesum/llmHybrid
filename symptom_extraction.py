"""
symptom_extraction.py - Symptom Extraction Module
Uses keyword matching + embedding similarity for symptom extraction.
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict

# Common mental health keywords for symptom extraction
SYMPTOM_KEYWORDS = {
    # Mood-related
    "sadness": ["sad", "sadness", "depressed", "feeling down", "blue", "unhappy", "miserable"],
    "loss of interest": ["lost interest", "no interest", "don't enjoy", "anhedonia", "stopped doing"],
    "hopelessness": ["hopeless", "no hope", "nothing to live for", "pointless", "giving up"],
    "guilt": ["guilty", "shame", "self-blame", "worthless", "bad person"],
    "worthlessness": ["worthless", "useless", "nothing good", "failure"],
    "irritability": ["irritable", "annoyed", "frustrated", "short-tempered", "angry easily"],
    
    # Anxiety-related
    "worry": ["worry", "worried", "anxious", "nervous", "concerned", "racing thoughts"],
    "restlessness": ["restless", "can't relax", "on edge", "tense", "jumpy"],
    "panic": ["panic", "panic attack", "terror", "overwhelming fear", "freaking out"],
    "fear": ["fear", "scared", "afraid", "phobia", "frightened"],
    "shortness of breath": ["shortness of breath", "can't breathe", "breathing fast", "hyperventilate"],
    "heart racing": ["heart racing", "heart pounding", "palpitations", "fast heartbeat"],
    
    # Sleep-related
    "sleep changes": ["sleep", "insomnia", "can't sleep", "too much sleep", "tired", "exhausted"],
    "nightmares": ["nightmare", "bad dreams", "terrifying dreams", "wake up scared"],
    "early waking": ["wake up early", "can't stay asleep", "middle of night"],
    
    # Energy-related
    "fatigue": ["fatigue", "tired", "exhausted", "no energy", "drained", "fatigue"],
    "low energy": ["low energy", "no motivation", "sluggish", "slow"],
    "high energy": ["high energy", "racing", "can't sit still", "too much energy"],
    
    # Cognitive
    "concentration problems": ["concentrate", "focus", "memory", "confused", "can't think"],
    "difficulty concentrating": ["distracted", "scatterbrained", "mind blank"],
    "racing thoughts": ["racing thoughts", "mind racing", "too many thoughts"],
    
    # Physical
    "appetite changes": ["appetite", "eating", "weight loss", "weight gain", "no appetite"],
    "weight changes": ["lost weight", "gained weight", "body changed"],
    "physical symptoms": ["headache", "stomach pain", "nausea", "aches", "pains"],
    
    # Behavioral
    "social withdrawal": ["withdrawn", "isolated", "no friends", "stay home", "avoid people"],
    "self-harm": ["self-harm", "cutting", "hurt myself", "suicide", "kill myself"],
    "suicidal thoughts": ["suicidal", "want to die", "end it all", "better off dead"],
    
    # Psychotic symptoms
    "hallucinations": ["hear voices", "seeing things", "hallucination", "voices in head"],
    "delusions": ["belief", "paranoid", "someone following", "plot against me"],
    
    # Trauma-related
    "flashbacks": ["flashback", "relive", "memories come back"],
    "intrusive thoughts": ["intrusive", "can't stop thinking", "obsessive thoughts"],
    "hypervigilance": ["on guard", "watchful", "jumpy", "startled"],
    "avoidance": ["avoid", "stay away from", "can't face"],
    
    # Mania/Bipolar
    "mood swings": ["mood swings", "up and down", "oscillate", "unstable"],
    "elevated mood": ["euphoric", "great", "on top of the world", "better than ever"],
    "mania": ["mania", "manic", "out of control", "wild"],
    "grandiosity": ["special", "better than others", "important"],
    "impulsivity": ["impulsive", "reckless", "spend money", "risky"],
    
    # OCD
    "obsessions": ["obsessed", "can't get out of mind", "fixated"],
    "compulsions": ["compulsion", "have to", "ritual", "repeat"],
    "checking": ["check", "double-check", "make sure"],
    "cleaning": ["clean", "wash", "sterile"],
    
    # ADHD
    "inattention": ["can't focus", "daydream", "don't listen", "space out"],
    "hyperactivity": ["fidget", "can't sit still", "bounce leg", "talk too much"],
    "impulsivity": ["interrupt", "act without thinking", "impulsive"],
    "forgetfulness": ["forget", "lost", "missed", "remember"],
    "fidgeting": ["fidget", "tap", "bounce", "restless hands"],
    
    # Eating
    "bingeing": ["binge", "eat too much", "out of control eating"],
    "purging": ["purge", "vomit", "throw up", "laxative"],
    "food restriction": ["restrict", "won't eat", "afraid to eat"],
    "body image concerns": ["body image", "fat", "ugly", "look different"],
    
    # Other
    "confusion": ["confused", "disoriented", "can't make decisions"],
    "flat affect": ["flat", "no expression", "emotionless"],
    "disorganized thinking": ["jumbled", "scattered", "can't organize thoughts"],
    "emotional numbness": ["numb", "empty", "dead inside", "feel nothing"]
}


class SymptomExtractor:
    """Extracts symptoms from user queries using keyword matching."""
    
    def __init__(self, keyword_dict: Dict = None):
        self.keyword_dict = keyword_dict or SYMPTOM_KEYWORDS
        # Create reverse mapping: keyword -> symptom
        self.reverse_map = {}
        for symptom, keywords in self.keyword_dict.items():
            for kw in keywords:
                self.reverse_map[kw.lower()] = symptom
    
    def extract_symptoms(self, query: str) -> List[str]:
        """
        Extract symptoms from user query using keyword matching.
        Returns list of extracted symptoms.
        """
        query_lower = query.lower()
        extracted = set()
        
        # Check each keyword in the query
        for keyword, symptom in self.reverse_map.items():
            if keyword in query_lower:
                extracted.add(symptom)
        
        # Also check for multi-word patterns
        multi_word_patterns = {
            "can't sleep": "sleep changes",
            "difficulty sleeping": "sleep changes",
            "trouble sleeping": "sleep changes",
            "sleep problems": "sleep changes",
            "loss of interest": "loss of interest",
            "lost interest": "loss of interest",
            "no interest": "loss of interest",
            "feelings of guilt": "guilt",
            "feel worthless": "worthlessness",
            "panic attack": "panic",
            "heart racing": "heart racing",
            "short of breath": "shortness of breath",
            "racing thoughts": "racing thoughts",
            "concentration issues": "concentration problems",
            "trouble focusing": "concentration problems",
            "want to die": "suicidal thoughts",
            "better off dead": "suicidal thoughts",
            "hurt myself": "self-harm",
            "cutting": "self-harm",
            "hear voices": "hallucinations",
            "see things": "hallucinations",
            "paranoid": "delusions",
            "someone following": "delusions",
            "relive": "flashbacks",
            "bad dreams": "nightmares",
            "mood swings": "mood swings",
            "up and down": "mood swings",
            "euphoric": "elevated mood",
            "on top of the world": "elevated mood",
            "obsessed": "obsessions",
            "compulsion": "compulsions",
            "have to": "compulsions",
            "ritual": "compulsions",
            "daydream": "inattention",
            "space out": "inattention",
            "fidget": "hyperactivity",
            "tap": "hyperactivity",
            "interrupt": "impulsivity",
            "binge": "bingeing",
            "throw up": "purging",
            "vomit": "purging",
            "restrict": "food restriction",
            "afraid to eat": "food restriction",
            "body image": "body image concerns",
            "feel empty": "emotional numbness",
            "feel numb": "emotional numbness"
        }
        
        for pattern, symptom in multi_word_patterns.items():
            if pattern in query_lower:
                extracted.add(symptom)
        
        return list(extracted)
    
    def extract_with_confidence(self, query: str) -> List[Tuple[str, float]]:
        """
        Extract symptoms with confidence scores based on keyword frequency.
        """
        query_lower = query.lower()
        symptom_counts = defaultdict(int)
        
        for keyword, symptom in self.reverse_map.items():
            if keyword in query_lower:
                symptom_counts[symptom] += 1
        
        # Convert counts to confidence scores
        results = []
        for symptom, count in symptom_counts.items():
            confidence = min(1.0, count * 0.5)  # Cap at 1.0
            results.append((symptom, confidence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)


def extract_symptoms_simple(query: str) -> List[str]:
    """Simple symptom extraction function."""
    extractor = SymptomExtractor()
    return extractor.extract_symptoms(query)


if __name__ == "__main__":
    # Test symptom extraction
    extractor = SymptomExtractor()
    
    test_queries = [
        "I feel so sad and lost interest in everything I used to enjoy",
        "I've been having panic attacks and can't sleep at night",
        "I hear voices in my head and think someone is following me",
        "I'm so irritable and can't concentrate on anything"
    ]
    
    print("Symptom Extraction Test:")
    print("=" * 60)
    for query in test_queries:
        symptoms = extractor.extract_symptoms(query)
        print(f"\nQuery: {query}")
        print(f"Extracted symptoms: {symptoms}")