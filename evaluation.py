"""
evaluation.py - Hallucination Evaluation Module
Measures hallucination rates before and after KG+SEAL system.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class EvaluationResult:
    """Stores evaluation metrics."""
    total_samples: int
    correct_predictions: int
    hallucinated_predictions: int
    abstained_predictions: int
    hallucination_rate: float
    accuracy: float
    abstention_rate: float


@dataclass
class SampleResult:
    """Stores result for a single sample."""
    query: str
    kg_prediction: str
    llm_prediction: str
    final_decision: str
    is_hallucination: bool
    risk_level: str


class HallucinationEvaluator:
    """
    Evaluates hallucination rates and system performance.
    Compares LLM-only vs KG+SEAL system.
    """
    
    def __init__(self, ground_truth: Dict[str, str] = None):
        """
        Initialize evaluator.
        
        Args:
            ground_truth: Dict mapping queries to correct disorders
        """
        self.ground_truth = ground_truth or {}
        self.results: List[SampleResult] = []
        self.llm_only_results: List[SampleResult] = []
        self.kg_seal_results: List[SampleResult] = []
    
    def add_sample(self, 
                   query: str,
                   kg_prediction: str,
                   llm_prediction: str,
                   final_decision: str,
                   is_hallucination: bool,
                   risk_level: str):
        """Add a sample result."""
        result = SampleResult(
            query=query,
            kg_prediction=kg_prediction,
            llm_prediction=llm_prediction,
            final_decision=final_decision,
            is_hallucination=is_hallucination,
            risk_level=risk_level
        )
        self.results.append(result)
        self.kg_seal_results.append(result)
    
    def add_llm_only_sample(self,
                           query: str,
                           llm_prediction: str,
                           is_hallucination: bool,
                           risk_level: str):
        """Add LLM-only sample for comparison."""
        result = SampleResult(
            query=query,
            kg_prediction="N/A",
            llm_prediction=llm_prediction,
            final_decision=llm_prediction,
            is_hallucination=is_hallucination,
            risk_level=risk_level
        )
        self.llm_only_results.append(result)
    
    def compute_hallucination_rate(self, results: List[SampleResult]) -> float:
        """Compute hallucination rate for given results."""
        if not results:
            return 0.0
        
        hallucinated = sum(1 for r in results if r.is_hallucination)
        return hallucinated / len(results)
    
    def evaluate(self) -> EvaluationResult:
        """Compute overall evaluation metrics."""
        if not self.kg_seal_results:
            return EvaluationResult(
                total_samples=0,
                correct_predictions=0,
                hallucinated_predictions=0,
                abstained_predictions=0,
                hallucination_rate=0.0,
                accuracy=0.0,
                abstention_rate=0.0
            )
        
        total = len(self.kg_seal_results)
        hallucinated = sum(1 for r in self.kg_seal_results if r.is_hallucination)
        abstained = sum(1 for r in self.kg_seal_results if r.final_decision == "ABSTAIN")
        
        # Correct = not hallucinated and not abstained
        correct = total - hallucinated - abstained
        
        return EvaluationResult(
            total_samples=total,
            correct_predictions=correct,
            hallucinated_predictions=hallucinated,
            abstained_predictions=abstained,
            hallucination_rate=hallucinated / total if total > 0 else 0.0,
            accuracy=correct / total if total > 0 else 0.0,
            abstention_rate=abstained / total if total > 0 else 0.0
        )
    
    def compare_systems(self) -> Dict:
        """Compare LLM-only vs KG+SEAL system."""
        llm_hallucination = self.compute_hallucination_rate(self.llm_only_results)
        kg_seal_hallucination = self.compute_hallucination_rate(self.kg_seal_results)
        
        llm_eval = self.evaluate_llm_only()
        kg_seal_eval = self.evaluate()
        
        return {
            "llm_only": {
                "hallucination_rate": llm_hallucination,
                "total_samples": len(self.llm_only_results),
                "accuracy": llm_eval.accuracy
            },
            "kg_seal": {
                "hallucination_rate": kg_seal_hallucination,
                "total_samples": len(self.kg_seal_results),
                "accuracy": kg_seal_eval.accuracy,
                "abstention_rate": kg_seal_eval.abstention_rate
            },
            "improvement": {
                "hallucination_reduction": llm_hallucination - kg_seal_hallucination,
                "hallucination_reduction_pct": (
                    (llm_hallucination - kg_seal_hallucination) / llm_hallucination * 100
                    if llm_hallucination > 0 else 0
                )
            }
        }
    
    def evaluate_llm_only(self) -> EvaluationResult:
        """Evaluate LLM-only results."""
        if not self.llm_only_results:
            return EvaluationResult(
                total_samples=0,
                correct_predictions=0,
                hallucinated_predictions=0,
                abstained_predictions=0,
                hallucination_rate=0.0,
                accuracy=0.0,
                abstention_rate=0.0
            )
        
        total = len(self.llm_only_results)
        hallucinated = sum(1 for r in self.llm_only_results if r.is_hallucination)
        
        return EvaluationResult(
            total_samples=total,
            correct_predictions=total - hallucinated,
            hallucinated_predictions=hallucinated,
            abstained_predictions=0,
            hallucination_rate=hallucinated / total if total > 0 else 0.0,
            accuracy=(total - hallucinated) / total if total > 0 else 0.0,
            abstention_rate=0.0
        )
    
    def get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels."""
        distribution = defaultdict(int)
        for result in self.kg_seal_results:
            distribution[result.risk_level] += 1
        return dict(distribution)
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON file."""
        data = {
            "comparison": self.compare_systems(),
            "risk_distribution": self.get_risk_distribution(),
            "samples": [
                {
                    "query": r.query,
                    "kg_prediction": r.kg_prediction,
                    "llm_prediction": r.llm_prediction,
                    "final_decision": r.final_decision,
                    "is_hallucination": r.is_hallucination,
                    "risk_level": r.risk_level
                }
                for r in self.kg_seal_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print evaluation summary."""
        comparison = self.compare_systems()
        
        print("\n" + "=" * 70)
        print("HALLUCINATION EVALUATION SUMMARY")
        print("=" * 70)
        
        print("\n📊 LLM-Only System:")
        print(f"   Total samples: {comparison['llm_only']['total_samples']}")
        print(f"   Hallucination rate: {comparison['llm_only']['hallucination_rate']:.2%}")
        print(f"   Accuracy: {comparison['llm_only']['accuracy']:.2%}")
        
        print("\n📊 KG+SEAL System:")
        print(f"   Total samples: {comparison['kg_seal']['total_samples']}")
        print(f"   Hallucination rate: {comparison['kg_seal']['hallucination_rate']:.2%}")
        print(f"   Accuracy: {comparison['kg_seal']['accuracy']:.2%}")
        print(f"   Abstention rate: {comparison['kg_seal']['abstention_rate']:.2%}")
        
        print("\n📈 Improvement:")
        print(f"   Hallucination reduction: {comparison['improvement']['hallucination_reduction']:.2%}")
        print(f"   Hallucination reduction %: {comparison['improvement']['hallucination_reduction_pct']:.1f}%")
        
        print("\n📋 Risk Distribution:")
        for risk, count in self.get_risk_distribution().items():
            print(f"   {risk.upper()}: {count}")


# Sample ground truth for testing
SAMPLE_GROUND_TRUTH = {
    "I feel so sad and lost interest in everything": "Depression",
    "I've been having panic attacks and can't breathe": "Panic Disorder",
    "I hear voices in my head telling me things": "Schizophrenia",
    "I can't sleep at night and keep having nightmares": "PTSD",
    "I'm so irritable and can't concentrate on anything": "ADHD",
    "I binge eat and then make myself vomit": "Eating Disorder",
    "I have obsessive thoughts and need to check things": "OCD",
    "My mood swings between very high and very low": "Bipolar Disorder",
    "I feel worried all the time and can't relax": "Anxiety Disorder",
    "I want to hurt myself": "Depression"
}


def create_evaluator(ground_truth: Dict = None) -> HallucinationEvaluator:
    """Factory function to create evaluator."""
    return HallucinationEvaluator(ground_truth=ground_truth or SAMPLE_GROUND_TRUTH)


if __name__ == "__main__":
    # Test evaluation
    print("Testing Hallucination Evaluator:")
    print("=" * 60)
    
    evaluator = create_evaluator()
    
    # Add sample results
    evaluator.add_sample(
        query="I feel so sad and lost interest in everything",
        kg_prediction="Depression",
        llm_prediction="Depression",
        final_decision="ANSWER",
        is_hallucination=False,
        risk_level="normal"
    )
    
    evaluator.add_sample(
        query="I hear voices in my head",
        kg_prediction="Schizophrenia",
        llm_prediction="Depression",  # Wrong!
        final_decision="ABSTAIN",
        is_hallucination=False,  # Abstained so not counted as hallucination
        risk_level="medium"
    )
    
    evaluator.add_llm_only_sample(
        query="I hear voices in my head",
        llm_prediction="Depression",  # Wrong!
        is_hallucination=True,
        risk_level="medium"
    )
    
    # Print summary
    evaluator.print_summary()