"""
seal.py - SEAL Module (Self-Evaluation and Abstention Layer)
Controls abstention based on agreement and confidence thresholds.
Includes rejection token mechanism for high-risk scenarios.
"""

from typing import Dict, Optional, Tuple
from enum import Enum


class RiskLevel(Enum):
    """Risk level classification."""
    NORMAL = "normal"
    MEDIUM = "medium"
    DANGER = "danger"


# Special tokens for LLM communication
class SEALTokens:
    """Special tokens for SEAL communication with LLM."""
    # Rejection tokens - sent to LLM when abstaining
    REJECT_TOKEN = "<|REJECT|>"
    ABSTAIN_TOKEN = "<|ABSTAIN|>"
    DANGER_TOKEN = "<|DANGER|>"
    
    # Safety tokens
    SAFE_TOKEN = "<|SAFE|>"
    UNCERTAIN_TOKEN = "<|UNCERTAIN|>"
    
    # Meta tokens
    SEPARATOR = "<|SEP|>"
    END = "<|END|>"


class SealModule:
    """
    SEAL Module - Controls when to abstain from answering.
    Implements safety checks, risk classification, and rejection token generation.
    
    When risk is HIGH (DANGER), the model sends a rejection token instead of
    generating a response. This ensures the LLM never produces potentially
    harmful content.
    """
    
    def __init__(self, 
                 agreement_threshold: float = 0.5,
                 confidence_threshold: float = 0.6,
                 danger_keywords: list = None,
                 use_rejection_tokens: bool = True):
        """
        Initialize SEAL module.
        
        Args:
            agreement_threshold: Minimum agreement score to answer
            confidence_threshold: Minimum confidence to answer
            danger_keywords: Keywords indicating danger level
            use_rejection_tokens: Whether to use rejection tokens for LLM
        """
        self.agreement_threshold = agreement_threshold
        self.confidence_threshold = confidence_threshold
        self.use_rejection_tokens = use_rejection_tokens
        
        # Default danger keywords
        self.danger_keywords = danger_keywords or [
            "suicide", "suicidal", "kill myself", "want to die", "end it all",
            "better off dead", "self-harm", "hurt myself", "cutting", "overdose",
            "hang myself", "jump off", "slit wrists", "kill me", "death wish",
            "abuse", "torture", "assault", "rape", "molestation"
        ]
        
        # Medium risk keywords
        self.medium_keywords = [
            "depressed", "depression", "anxiety", "panic", "trauma", "PTSD",
            "self harm", "hurt", "empty", "hopeless", "worthless", "alone",
            "nightmare", "flashback", "avoid", "fear", "worry", "stress"
        ]
    
    def classify_risk(self, query: str) -> Tuple[RiskLevel, str]:
        """
        Classify input query into risk level.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (risk_level, reason)
        """
        query_lower = query.lower()
        
        # Check for danger keywords
        for keyword in self.danger_keywords:
            if keyword in query_lower:
                return RiskLevel.DANGER, f"Danger keyword detected: {keyword}"
        
        # Check for medium risk keywords
        for keyword in self.medium_keywords:
            if keyword in query_lower:
                return RiskLevel.MEDIUM, f"Medium risk keyword detected: {keyword}"
        
        return RiskLevel.NORMAL, "No risk keywords detected"
    
    def should_abstain(self, 
                       agreement_score: float,
                       confidence: float,
                       risk_level: RiskLevel) -> Tuple[bool, str]:
        """
        Determine if system should abstain from answering.
        
        Args:
            agreement_score: Agreement score between KG and LLM
            confidence: Final confidence score
            risk_level: Classified risk level
            
        Returns:
            Tuple of (should_abstain, reason)
        """
        # Always abstain for danger level
        if risk_level == RiskLevel.DANGER:
            return True, "Danger level input - must refer to professional"
        
        # Check agreement threshold
        if agreement_score < self.agreement_threshold:
            return True, f"Low agreement score: {agreement_score:.2f} < {self.agreement_threshold}"
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return True, f"Low confidence: {confidence:.2f} < {self.confidence_threshold}"
        
        return False, "Prediction meets safety criteria"
    
    def get_rejection_token(self, risk_level: RiskLevel) -> str:
        """
        Get the appropriate rejection token based on risk level.
        
        When risk is HIGH/DANGER, the model sends a rejection token instead
        of generating a response. This prevents the LLM from producing
        potentially harmful content.
        
        Args:
            risk_level: The classified risk level
            
        Returns:
            Rejection token string
        """
        if not self.use_rejection_tokens:
            return ""
        
        if risk_level == RiskLevel.DANGER:
            return SEALTokens.DANGER_TOKEN
        elif risk_level == RiskLevel.MEDIUM:
            return SEALTokens.ABSTAIN_TOKEN
        else:
            return SEALTokens.REJECT_TOKEN
    
    def should_send_rejection_token(self, 
                                     agreement_score: float,
                                     confidence: float,
                                     risk_level: RiskLevel) -> Tuple[bool, str]:
        """
        Determine if a rejection token should be sent to the LLM.
        
        This is the primary interface for LLM integration. When True is returned,
        the LLM should receive a rejection token instead of generating a response.
        
        Args:
            agreement_score: Agreement score between KG and LLM
            confidence: Final confidence score
            risk_level: Classified risk level
            
        Returns:
            Tuple of (should_reject, rejection_token)
        """
        # High risk always gets rejection token
        if risk_level == RiskLevel.DANGER:
            return True, self.get_rejection_token(RiskLevel.DANGER)
        
        # Low agreement gets rejection token
        if agreement_score < self.agreement_threshold:
            return True, SEALTokens.REJECT_TOKEN
        
        # Low confidence gets rejection token
        if confidence < self.confidence_threshold:
            return True, SEALTokens.ABSTAIN_TOKEN
        
        return False, ""
    
    def format_llm_prompt_with_seal(self, 
                                    base_prompt: str,
                                    agreement_score: float,
                                    confidence: float,
                                    risk_level: RiskLevel) -> str:
        """
        Format the LLM prompt with SEAL context.
        
        Injects SEAL metadata into the prompt so the LLM knows
        whether to respond or abstain.
        
        Args:
            base_prompt: Original user prompt
            agreement_score: KG-LLM agreement score
            confidence: Confidence score
            risk_level: Risk level
            
        Returns:
            Formatted prompt with SEAL context
        """
        should_reject, token = self.should_send_rejection_token(
            agreement_score, confidence, risk_level
        )
        
        if should_reject:
            # Inject rejection token into prompt
            seal_context = f"\n{SEALTokens.SEPARATOR}\n"
            seal_context += f"SAFETY STATUS: {token}\n"
            seal_context += "ACTION: Do not generate a response. "
            seal_context += "Return the rejection token only.\n"
            seal_context += f"{SEALTokens.SEPARATOR}\n"
            
            return base_prompt + seal_context
        else:
            # Normal processing - add safety context
            seal_context = f"\n{SEALTokens.SEPARATOR}\n"
            seal_context += f"SAFETY STATUS: {SEALTokens.SAFE_TOKEN}\n"
            seal_context += f"AGREEMENT SCORE: {agreement_score:.2f}\n"
            seal_context += f"CONFIDENCE: {confidence:.2f}\n"
            seal_context += f"{SEALTokens.SEPARATOR}\n"
            
            return base_prompt + seal_context
    
    def validate_output(self, 
                       prediction: Dict,
                       risk_level: RiskLevel) -> Tuple[bool, str, str]:
        """
        Validate final output before returning to user.
        
        Args:
            prediction: Final prediction dict
            risk_level: Risk level of input
            
        Returns:
            Tuple of (is_valid, decision, message)
        """
        # For danger level, always use safe fallback
        if risk_level == RiskLevel.DANGER:
            return False, "ABSTAIN", "Please consult a mental health professional immediately."
        
        # Check prediction quality
        decision = prediction.get("decision", "ABSTAIN")
        confidence = prediction.get("final_confidence", 0)
        agreement = prediction.get("agreement_score", 0)
        
        if decision == "ABSTAIN":
            return False, "ABSTAIN", "Please consult a professional for proper evaluation."
        
        if confidence < self.confidence_threshold:
            return False, "ABSTAIN", "Confidence too low. Please consult a professional."
        
        if agreement < self.agreement_threshold:
            return False, "CAUTIOUS", "Low agreement between systems. Please verify with a professional."
        
        return True, decision, ""
    
    def get_safe_message(self, 
                        decision: str, 
                        disorder: Optional[str] = None,
                        risk_level: RiskLevel = RiskLevel.NORMAL) -> str:
        """
        Generate safe response message.
        
        Args:
            decision: Final decision (ANSWER, CAUTIOUS ANSWER, ABSTAIN)
            disorder: Predicted disorder (if any)
            risk_level: Risk level
            
        Returns:
            Safe response string
        """
        if risk_level == RiskLevel.DANGER:
            return (
                "I'm concerned about what you're sharing. "
                "If you're having thoughts of harming yourself, "
                "please reach out for immediate help:\n"
                "- National Suicide Prevention Lifeline: 988\n"
                "- Crisis Text Line: Text HOME to 741741\n"
                "- Emergency services: 911\n\n"
                "Please consult a mental health professional for proper evaluation."
            )
        
        if decision == "ABSTAIN":
            return (
                "I'm not able to provide a reliable assessment based on the information provided. "
                "Please consult a mental health professional for proper evaluation. "
                "If you're in crisis, please call 988 (Suicide Prevention Lifeline)."
            )
        
        if decision == "CAUTIOUS ANSWER" and disorder:
            return (
                f"Based on the analysis, the most likely condition may be: {disorder}\n\n"
                "⚠️ IMPORTANT: This is NOT a medical diagnosis. "
                "Please consult a mental health professional for proper evaluation.\n"
                "If you're experiencing distress, reach out to a qualified provider."
            )
        
        if disorder:
            return (
                f"Based on the analysis, the most likely condition is: {disorder}\n\n"
                "⚠️ IMPORTANT: This is NOT a medical diagnosis. "
                "This system is for educational/research purposes only. "
                "Please consult a mental health professional for proper evaluation."
            )
        
        return "Please consult a mental health professional for proper evaluation."
    
    def get_disclaimer(self) -> str:
        """Get standard disclaimer text."""
        return (
            "⚠️ DISCLAIMER: This system is for educational and research purposes only. "
            "It is NOT a medical diagnosis tool. "
            "The predictions made by this system should NOT be used as a substitute "
            "for professional medical advice, diagnosis, or treatment. "
            "Always seek the advice of qualified mental health professionals "
            "for any questions about mental health conditions."
        )


def create_seal_module(agreement_threshold: float = 0.5,
                       confidence_threshold: float = 0.6) -> SealModule:
    """Factory function to create SEAL module."""
    return SealModule(
        agreement_threshold=agreement_threshold,
        confidence_threshold=confidence_threshold
    )


if __name__ == "__main__":
    # Test SEAL module
    print("Testing SEAL Module:")
    print("=" * 60)
    
    seal = SealModule()
    
    # Test risk classification
    test_queries = [
        "I feel so sad and lost interest in everything",
        "I want to kill myself",
        "I've been having panic attacks"
    ]
    
    print("\nRisk Classification:")
    for query in test_queries:
        risk, reason = seal.classify_risk(query)
        print(f"Query: {query}")
        print(f"Risk: {risk.value} - {reason}")
        print()
    
    # Test abstention
    print("\nAbstention Test:")
    should_abstain, reason = seal.should_abstain(0.3, 0.5, RiskLevel.NORMAL)
    print(f"Agreement: 0.3, Confidence: 0.5, Risk: NORMAL")
    print(f"Should abstain: {should_abstain}, Reason: {reason}")