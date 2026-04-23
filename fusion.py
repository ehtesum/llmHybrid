"""
fusion.py - Agreement & Fusion Module
Compares KG-based and LLM-based predictions and computes agreement score.
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum


class AgreementLevel(Enum):
    """Agreement levels between KG and LLM predictions."""
    FULL_MATCH = "full_match"
    PARTIAL_MATCH = "partial_match"
    NO_MATCH = "no_match"


class FusionModule:
    """
    Fusion module that combines KG and LLM predictions.
    Implements decision rules for final output.
    """
    
    def __init__(self, 
                 full_match_threshold: float = 0.8,
                 partial_match_threshold: float = 0.5,
                 min_confidence: float = 0.6):
        """
        Initialize fusion module.
        
        Args:
            full_match_threshold: Threshold for full agreement
            partial_match_threshold: Threshold for partial agreement
            min_confidence: Minimum confidence to accept prediction
        """
        self.full_match_threshold = full_match_threshold
        self.partial_match_threshold = partial_match_threshold
        self.min_confidence = min_confidence
    
    def compute_agreement(self, 
                         kg_prediction: Tuple[str, float],
                         llm_prediction: Tuple[str, float]) -> Tuple[float, AgreementLevel]:
        """
        Compute agreement between KG and LLM predictions.
        
        Args:
            kg_prediction: (disorder, confidence) from KG
            llm_prediction: (disorder, confidence) from LLM
            
        Returns:
            Tuple of (agreement_score, agreement_level)
        """
        kg_disorder, kg_conf = kg_prediction
        llm_disorder, llm_conf = llm_prediction
        
        # Exact match
        if kg_disorder.lower() == llm_disorder.lower():
            # Consider confidence levels
            conf_diff = abs(kg_conf - llm_conf)
            if conf_diff < 0.2:
                return 1.0, AgreementLevel.FULL_MATCH
            else:
                return 0.8, AgreementLevel.FULL_MATCH
        
        # Check for partial matches (related disorders)
        related_disorders = self._get_related_disorders(kg_disorder)
        if llm_disorder in related_disorders:
            return 0.6, AgreementLevel.PARTIAL_MATCH
        
        # Check symptom overlap
        return 0.2, AgreementLevel.NO_MATCH
    
    def _get_related_disorders(self, disorder: str) -> List[str]:
        """Get related disorders for partial matching."""
        related_map = {
            "Depression": ["Depression", "Dysthymia", "Major Depressive Disorder"],
            "Anxiety Disorder": ["Anxiety Disorder", "Generalized Anxiety", "Panic Disorder"],
            "Panic Disorder": ["Panic Disorder", "Anxiety Disorder", "Agoraphobia"],
            "Bipolar Disorder": ["Bipolar Disorder", "Cyclothymia", "Bipolar II"],
            "Schizophrenia": ["Schizophrenia", "Schizoaffective", "Psychosis"],
            "PTSD": ["PTSD", "Acute Stress Disorder", "Adjustment Disorder"],
            "OCD": ["OCD", "Anxiety Disorder", "Hoarding Disorder"],
            "ADHD": ["ADHD", "Attention Deficit", "Hyperactivity"],
            "Eating Disorder": ["Eating Disorder", "Anorexia", "Bulimia", "Binge Eating"],
            "Insomnia": ["Insomnia", "Sleep Disorder", "Sleep Apnea"]
        }
        return related_map.get(disorder, [disorder])
    
    def fuse(self, 
             kg_result: Dict,
             llm_result: Dict) -> Dict:
        """
        Fuse KG and LLM results to produce final decision.
        
        Args:
            kg_result: Dict with 'disorder', 'confidence', 'symptoms'
            llm_result: Dict with 'disorder', 'confidence', 'reasoning'
            
        Returns:
            Dict with final decision, agreement score, and metadata
        """
        kg_prediction = (kg_result.get("disorder", ""), kg_result.get("confidence", 0))
        llm_prediction = (llm_result.get("disorder", ""), llm_result.get("confidence", 0))
        
        agreement_score, agreement_level = self.compute_agreement(
            kg_prediction, llm_prediction
        )
        
        # Decision rules
        if agreement_level == AgreementLevel.FULL_MATCH:
            if kg_prediction[1] >= self.min_confidence and llm_prediction[1] >= self.min_confidence:
                decision = "ANSWER"
                final_disorder = kg_prediction[0]
                final_confidence = (kg_prediction[1] + llm_prediction[1]) / 2
            else:
                decision = "CAUTIOUS ANSWER"
                final_disorder = kg_prediction[0]
                final_confidence = min(kg_prediction[1], llm_prediction[1])
        
        elif agreement_level == AgreementLevel.PARTIAL_MATCH:
            decision = "CAUTIOUS ANSWER"
            # Prefer higher confidence
            if kg_prediction[1] >= llm_prediction[1]:
                final_disorder = kg_prediction[0]
            else:
                final_disorder = llm_prediction[0]
            final_confidence = min(kg_prediction[1], llm_prediction[1]) * 0.7
        
        else:  # NO_MATCH
            # Check if either has high confidence
            if kg_prediction[1] >= 0.8 or llm_prediction[1] >= 0.8:
                decision = "CAUTIOUS ANSWER"
                final_disorder = kg_prediction[0] if kg_prediction[1] > llm_prediction[1] else llm_prediction[0]
                final_confidence = max(kg_prediction[1], llm_prediction[1]) * 0.5
            else:
                decision = "ABSTAIN"
                final_disorder = None
                final_confidence = 0.0
        
        return {
            "decision": decision,
            "final_disorder": final_disorder,
            "final_confidence": final_confidence,
            "agreement_score": agreement_score,
            "agreement_level": agreement_level.value,
            "kg_prediction": {
                "disorder": kg_prediction[0],
                "confidence": kg_prediction[1]
            },
            "llm_prediction": {
                "disorder": llm_prediction[0],
                "confidence": llm_prediction[1]
            }
        }
    
    def get_safe_fallback(self, decision: str) -> str:
        """Get safe fallback message based on decision."""
        fallbacks = {
            "ANSWER": "Based on the analysis, the most likely condition is: ",
            "CAUTIOUS ANSWER": "Based on available information, the possible condition is: ",
            "ABSTAIN": "Please consult a mental health professional for proper evaluation."
        }
        return fallbacks.get(decision, "Please consult a professional.")


def compute_agreement_score(kg_disorder: str, llm_disorder: str) -> float:
    """Simple function to compute agreement score."""
    if kg_disorder.lower() == llm_disorder.lower():
        return 1.0
    return 0.0


if __name__ == "__main__":
    # Test fusion module
    print("Testing Fusion Module:")
    print("=" * 60)
    
    fusion = FusionModule()
    
    # Test case 1: Full match
    kg_result = {"disorder": "Depression", "confidence": 0.85, "symptoms": ["sadness", "fatigue"]}
    llm_result = {"disorder": "Depression", "confidence": 0.80, "reasoning": "Symptom pattern matches"}
    
    result = fusion.fuse(kg_result, llm_result)
    print(f"\nTest 1 - Full Match:")
    print(f"KG: {kg_result['disorder']} ({kg_result['confidence']})")
    print(f"LLM: {llm_result['disorder']} ({llm_result['confidence']})")
    print(f"Result: {result}")
    
    # Test case 2: No match
    kg_result2 = {"disorder": "Depression", "confidence": 0.85, "symptoms": ["sadness"]}
    llm_result2 = {"disorder": "Anxiety Disorder", "confidence": 0.75, "reasoning": "Worry pattern"}
    
    result2 = fusion.fuse(kg_result2, llm_result2)
    print(f"\nTest 2 - No Match:")
    print(f"KG: {kg_result2['disorder']} ({kg_result2['confidence']})")
    print(f"LLM: {llm_result2['disorder']} ({llm_result2['confidence']})")
    print(f"Result: {result2}")