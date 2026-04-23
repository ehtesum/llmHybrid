"""
llm_qwen.py - Qwen LLM Module
Uses HuggingFace transformers for Qwen model inference.
Integrates with SEAL module for rejection token handling.
"""

from typing import Dict, List, Tuple, Optional
import warnings

# Try to import transformers, handle gracefully if not available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not installed. LLM functionality will be simulated.")

# Import SEAL tokens for rejection handling
try:
    from seal import SEALTokens, RiskLevel
except ImportError:
    # Fallback if seal module not available
    class SEALTokens:
        REJECT_TOKEN = "<|REJECT|>"
        ABSTAIN_TOKEN = "<|ABSTAIN|>"
        DANGER_TOKEN = "<|DANGER|>"
        SAFE_TOKEN = "<|SAFE|>"
    
    class RiskLevel:
        NORMAL = "normal"
        MEDIUM = "medium"
        DANGER = "danger"

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Default prompt template for symptom analysis
DEFAULT_PROMPT_TEMPLATE = """You are a mental health assistant. Given the following symptoms, 
identify the most likely mental health condition.

Symptoms: {symptoms}

Based on the symptoms provided, what is the most relevant mental health condition? 
Respond with only the disorder name and a confidence score (0-1).

Format your response as:
Disorder: [name]
Confidence: [0.0-1.0]
Reasoning: [brief explanation]

Remember: This is for educational purposes only and is not a medical diagnosis."""


class QwenLLM:
    """
    Qwen LLM for neural-based disorder prediction.
    
    Integrates with SEAL module to handle rejection tokens:
    - When SEAL signals high risk, LLM receives rejection token
    - LLM should not generate response when rejection token present
    - Returns safe fallback instead of model-generated content
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B", device: str = "auto"):
        """
        Initialize Qwen model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu", "cuda", "auto")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            print("Warning: transformers not available. Using simulated mode.")
    
    def _load_model(self):
        """Load Qwen model and tokenizer."""
        try:
            print(f"Loading Qwen model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            self.is_loaded = True
            print("Qwen model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load Qwen model: {e}")
            print("Using simulated mode for demonstration.")
            self.is_loaded = False
    
    def predict(self, 
                symptoms: List[str], 
                prompt_template: str = None,
                seal_context: dict = None) -> Tuple[str, float, str]:
        """
        Predict disorder from symptoms using LLM.
        
        Args:
            symptoms: List of extracted symptoms
            prompt_template: Custom prompt template
            seal_context: Optional SEAL context with rejection info
            
        Returns:
            Tuple of (predicted_disorder, confidence, reasoning)
        """
        # Check if SEAL sent rejection token
        if seal_context and seal_context.get("should_reject", False):
            rejection_token = seal_context.get("rejection_token", "")
            return self._handle_rejection(rejection_token, seal_context)
        
        prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE
        symptoms_str = ", ".join(symptoms)
        full_prompt = prompt.format(symptoms=symptoms_str)
        
        if self.is_loaded and self.model is not None:
            return self._predict_real(symptoms, full_prompt)
        else:
            return self._predict_simulated(symptoms)
    
    def _handle_rejection(self, rejection_token: str, seal_context: dict) -> Tuple[str, float, str]:
        """
        Handle rejection token from SEAL.
        
        When SEAL determines high risk, this method ensures the LLM
        does not generate a response. Instead, it returns a safe fallback.
        
        Args:
            rejection_token: The rejection token from SEAL
            seal_context: Full SEAL context
            
        Returns:
            Tuple of (disorder, confidence, reasoning) indicating rejection
        """
        reason = seal_context.get("reason", "Safety threshold not met")
        
        # Return rejection indicator instead of prediction
        return (
            f"[REJECTED: {rejection_token}]",
            0.0,
            f"Response blocked by SEAL. Reason: {reason}"
        )
    
    def check_and_handle_rejection(self, 
                                   base_prompt: str,
                                   agreement_score: float,
                                   confidence: float,
                                   risk_level: str) -> Tuple[bool, str]:
        """
        Check if prompt should be rejected before LLM generation.
        
        This is called before generating any LLM response to ensure
        safety checks are applied.
        
        Args:
            base_prompt: Original user prompt
            agreement_score: KG-LLM agreement score
            confidence: Confidence score
            risk_level: Risk level string
            
        Returns:
            Tuple of (should_reject, modified_prompt_or_token)
        """
        # Map risk level string to enum for comparison
        risk_mapping = {
            "danger": "DANGER",
            "medium": "MEDIUM", 
            "normal": "NORMAL"
        }
        risk_enum = risk_mapping.get(risk_level.lower(), "NORMAL")
        
        # Check rejection conditions
        if risk_enum == "DANGER":
            return True, SEALTokens.DANGER_TOKEN
        
        if agreement_score < 0.5:
            return True, SEALTokens.REJECT_TOKEN
        
        if confidence < 0.6:
            return True, SEALTokens.ABSTAIN_TOKEN
        
        # No rejection needed
        return False, base_prompt
    
    def _predict_real(self, symptoms: List[str], prompt: str) -> Tuple[str, float, str]:
        """Real prediction using loaded Qwen model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_response(response)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return self._predict_simulated(symptoms)
    
    def _predict_simulated(self, symptoms: List[str]) -> Tuple[str, float, str]:
        """
        Simulated prediction for demonstration when model is not available.
        Uses rule-based matching for demonstration.
        """
        # Map symptoms to likely disorders (simulated LLM output)
        symptom_to_disorder = {
            "sadness": ("Depression", 0.85),
            "loss of interest": ("Depression", 0.80),
            "hopelessness": ("Depression", 0.90),
            "worry": ("Anxiety Disorder", 0.75),
            "restlessness": ("Anxiety Disorder", 0.70),
            "panic": ("Panic Disorder", 0.85),
            "fear": ("Anxiety Disorder", 0.70),
            "flashbacks": ("PTSD", 0.90),
            "nightmares": ("PTSD", 0.75),
            "hallucinations": ("Schizophrenia", 0.95),
            "delusions": ("Schizophrenia", 0.90),
            "mood swings": ("Bipolar Disorder", 0.80),
            "elevated mood": ("Bipolar Disorder", 0.85),
            "mania": ("Bipolar Disorder", 0.90),
            "obsessions": ("OCD", 0.90),
            "compulsions": ("OCD", 0.85),
            "inattention": ("ADHD", 0.75),
            "hyperactivity": ("ADHD", 0.70),
            "bingeing": ("Eating Disorder", 0.85),
            "purging": ("Eating Disorder", 0.90),
            "suicidal thoughts": ("Depression", 0.95),
            "self-harm": ("Depression", 0.85),
            "sleep changes": ("Depression", 0.60),
            "fatigue": ("Depression", 0.55),
            "irritability": ("Depression", 0.50),
            "concentration problems": ("ADHD", 0.55),
            "intrusive thoughts": ("OCD", 0.75),
            "hypervigilance": ("PTSD", 0.80),
            "social withdrawal": ("Schizophrenia", 0.65),
            "emotional numbness": ("PTSD", 0.70)
        }
        
        # Aggregate scores
        disorder_scores = {}
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in symptom_to_disorder:
                disorder, score = symptom_to_disorder[symptom_lower]
                if disorder not in disorder_scores:
                    disorder_scores[disorder] = []
                disorder_scores[disorder].append(score)
        
        if not disorder_scores:
            return ("Unknown", 0.3, "No clear pattern detected from symptoms")
        
        # Average scores
        best_disorder = max(
            disorder_scores.keys(),
            key=lambda d: sum(disorder_scores[d]) / len(disorder_scores[d])
        )
        avg_score = sum(disorder_scores[best_disorder]) / len(disorder_scores[best_disorder])
        
        reasoning = f"Based on symptom pattern: {', '.join(symptoms[:3])}"
        return (best_disorder, min(0.95, avg_score + 0.1), reasoning)
    
    def _parse_response(self, response: str) -> Tuple[str, float, str]:
        """Parse LLM response to extract disorder, confidence, and reasoning."""
        lines = response.split('\n')
        disorder = "Unknown"
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            if line.startswith("Disorder:"):
                disorder = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return disorder, confidence, reasoning
    
    def get_confidence_from_logprobs(self, response: str) -> float:
        """
        Estimate confidence from log probabilities (if available).
        This is a placeholder - actual implementation would use model logits.
        """
        # Placeholder: return heuristic confidence
        return 0.7


def create_llm(model_name: str = None) -> QwenLLM:
    """Factory function to create LLM instance."""
    return QwenLLM(model_name=model_name or "Qwen/Qwen2-0.5B")


if __name__ == "__main__":
    # Test the LLM
    print("Testing Qwen LLM Module:")
    print("=" * 60)
    
    llm = QwenLLM()
    test_symptoms = ["sadness", "loss of interest", "fatigue", "sleep changes"]
    
    disorder, confidence, reasoning = llm.predict(test_symptoms)
    print(f"\nInput symptoms: {test_symptoms}")
    print(f"Predicted disorder: {disorder}")
    print(f"Confidence: {confidence}")
    print(f"Reasoning: {reasoning}")