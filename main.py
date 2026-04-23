"""
main.py - Main Pipeline
Hybrid KG + LLM (Qwen) System for Hallucination Reduction with SEAL Module

This is the main entry point that orchestrates the full pipeline:
1. Symptom Extraction
2. Knowledge Graph Query
3. LLM Prediction (Qwen)
4. Fusion & Agreement
5. SEAL Abstention Control
6. Safe Response Generation

Designed for Google Colab - works seamlessly in browser-based environment.

Usage:
    python main.py                    # Run default pipeline
    python main.py --interactive      # Interactive mode
    python main.py --evaluate         # Run evaluation
    python main.py --use_qwen         # Use actual Qwen model (GPU recommended)
    python main.py -i -e              # Combined options

Colab Usage:
    # Upload files or clone repo, then:
    !python main.py --evaluate
"""

import json
import sys
import os
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Colab detection
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Import all modules
from kg import KnowledgeGraph
from symptom_extraction import SymptomExtractor
from llm_qwen import QwenLLM
from fusion import FusionModule
from seal import SealModule, RiskLevel
from evaluation import HallucinationEvaluator, create_evaluator


# Sample user queries for testing
SAMPLE_QUERIES = [
    # Normal queries
    "I feel so sad and lost interest in everything I used to enjoy",
    "I've been having panic attacks and can't breathe properly",
    "I can't sleep at night and keep having bad dreams about the accident",
    "I'm so irritable and can't concentrate on anything lately",
    "I feel worried all the time and can't relax",
    
    # Medium risk queries
    "I hear voices in my head telling me things that aren't real",
    "I have obsessive thoughts and need to check the lock 10 times",
    "My mood swings between feeling amazing and feeling terrible",
    "I binge eat and then make myself vomit to lose weight",
    "I feel empty and like nothing matters anymore",
    
    # Danger queries
    "I want to kill myself",
    "I have thoughts about hurting myself",
    "I think I'm better off dead"
]


@dataclass
class PipelineResult:
    """Result from the full pipeline."""
    query: str
    extracted_symptoms: List[str]
    kg_result: Dict
    llm_result: Dict
    fusion_result: Dict
    risk_level: RiskLevel
    final_response: str
    should_abstain: bool


class HybridKGLLMPipeline:
    """
    Main pipeline that combines KG and LLM for mental health query analysis.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2-0.5B",
                 agreement_threshold: float = 0.5,
                 confidence_threshold: float = 0.6):
        """
        Initialize the pipeline.
        
        Args:
            model_name: Qwen model name from HuggingFace
            agreement_threshold: Minimum agreement for answer
            confidence_threshold: Minimum confidence for answer
        """
        print("Initializing Hybrid KG + LLM Pipeline...")
        print("=" * 70)
        
        # Initialize components
        self.kg = KnowledgeGraph()
        print("✓ Knowledge Graph loaded")
        
        self.symptom_extractor = SymptomExtractor()
        print("✓ Symptom Extractor initialized")
        
        self.llm = QwenLLM(model_name=model_name)
        print("✓ Qwen LLM initialized")
        
        self.fusion = FusionModule(
            full_match_threshold=0.8,
            partial_match_threshold=0.5,
            min_confidence=confidence_threshold
        )
        print("✓ Fusion Module initialized")
        
        self.seal = SealModule(
            agreement_threshold=agreement_threshold,
            confidence_threshold=confidence_threshold
        )
        print("✓ SEAL Module initialized")
        
        self.evaluator = create_evaluator()
        print("✓ Evaluation Module initialized")
        
        print("\n" + "=" * 70)
        print("Pipeline ready!")
        print("=" * 70)
    
    def process_query(self, query: str, verbose: bool = True) -> PipelineResult:
        """
        Process a single query through the full pipeline.
        
        Args:
            query: User query string
            verbose: Whether to print detailed output
            
        Returns:
            PipelineResult with all outputs
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing query: {query}")
            print(f"{'='*70}")
        
        # STAGE 1: Symptom Extraction
        if verbose:
            print("\n📌 STAGE 1: Symptom Extraction")
        symptoms = self.symptom_extractor.extract_symptoms(query)
        if verbose:
            print(f"   Extracted symptoms: {symptoms}")
        
        # STAGE 2: KG Query (Path A - Symbolic)
        if verbose:
            print("\n📌 STAGE 2a: KG Query (Symbolic Path)")
        kg_results = self.kg.query(symptoms, top_k=3)
        kg_disorder = kg_results[0][0] if kg_results else "Unknown"
        kg_confidence = kg_results[0][1] if kg_results else 0.0
        kg_result = {
            "disorder": kg_disorder,
            "confidence": kg_confidence,
            "symptoms": symptoms,
            "top_matches": kg_results
        }
        if verbose:
            print(f"   KG Prediction: {kg_disorder} (confidence: {kg_confidence:.2f})")
            if len(kg_results) > 1:
                print(f"   Other candidates: {kg_results[1:]}")
        
        # STAGE 2: LLM Prediction (Path B - Neural)
        if verbose:
            print("\n📌 STAGE 2b: LLM Prediction (Neural Path)")
        llm_disorder, llm_confidence, llm_reasoning = self.llm.predict(symptoms)
        llm_result = {
            "disorder": llm_disorder,
            "confidence": llm_confidence,
            "reasoning": llm_reasoning
        }
        if verbose:
            print(f"   LLM Prediction: {llm_disorder} (confidence: {llm_confidence:.2f})")
            print(f"   Reasoning: {llm_reasoning}")
        
        # STAGE 3: Fusion & Agreement
        if verbose:
            print("\n📌 STAGE 3: Fusion & Agreement")
        fusion_result = self.fusion.fuse(kg_result, llm_result)
        if verbose:
            print(f"   Agreement Score: {fusion_result['agreement_score']:.2f}")
            print(f"   Agreement Level: {fusion_result['agreement_level']}")
            print(f"   Decision: {fusion_result['decision']}")
        
        # Risk Classification
        if verbose:
            print("\n📌 Risk Classification")
        risk_level, risk_reason = self.seal.classify_risk(query)
        if verbose:
            print(f"   Risk Level: {risk_level.value.upper()}")
            print(f"   Reason: {risk_reason}")
        
        # SEAL Abstention Check
        should_abstain, abstain_reason = self.seal.should_abstain(
            fusion_result['agreement_score'],
            fusion_result['final_confidence'],
            risk_level
        )
        
        # Generate final response
        final_response = self.seal.get_safe_message(
            fusion_result['decision'] if not should_abstain else "ABSTAIN",
            fusion_result['final_disorder'],
            risk_level
        )
        
        # Add disclaimer
        final_response += "\n\n" + self.seal.get_disclaimer()
        
        if verbose:
            print("\n📌 Final Output:")
            print(f"   Decision: {fusion_result['decision'] if not should_abstain else 'ABSTAIN'}")
            print(f"   Predicted Disorder: {fusion_result['final_disorder']}")
            print(f"   Final Confidence: {fusion_result['final_confidence']:.2f}")
            print(f"   Should Abstain: {should_abstain}")
            if should_abstain:
                print(f"   Reason: {abstain_reason}")
        
        return PipelineResult(
            query=query,
            extracted_symptoms=symptoms,
            kg_result=kg_result,
            llm_result=llm_result,
            fusion_result=fusion_result,
            risk_level=risk_level,
            final_response=final_response,
            should_abstain=should_abstain
        )
    
    def process_batch(self, 
                     queries: List[str], 
                     verbose: bool = True,
                     show_progress: bool = True) -> List[PipelineResult]:
        """
        Process multiple queries.
        
        Args:
            queries: List of user queries
            verbose: Whether to print detailed output
            show_progress: Whether to show progress bar
            
        Returns:
            List of PipelineResults
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(queries, desc="Processing queries") if show_progress else queries
        except ImportError:
            iterator = queries
            if verbose:
                print("tqdm not available, skipping progress bar")
        
        results = []
        for query in iterator:
            result = self.process_query(query, verbose=verbose)
            results.append(result)
        
        return results
    
    def run_evaluation(self, queries: List[str] = None):
        """
        Run evaluation on sample queries.
        
        Args:
            queries: List of queries to evaluate
        """
        queries = queries or SAMPLE_QUERIES
        
        print("\n" + "=" * 70)
        print("RUNNING EVALUATION")
        print("=" * 70)
        
        # Process all queries
        for query in queries:
            result = self.process_query(query, verbose=False)
            
            # Determine if hallucination (for demo, we'll simulate)
            # In real evaluation, compare against ground truth
            is_hallucination = (
                result.fusion_result.get("decision") == "ANSWER" and
                result.kg_result["disorder"] != result.llm_result["disorder"]
            )
            
            # Add to evaluator
            self.evaluator.add_sample(
                query=query,
                kg_prediction=result.kg_result["disorder"],
                llm_prediction=result.llm_result["disorder"],
                final_decision=result.fusion_result["decision"],
                is_hallucination=is_hallucination,
                risk_level=result.risk_level.value
            )
        
        # Print evaluation summary
        self.evaluator.print_summary()
        
        return self.evaluator
    
    def save_results(self, results: List[PipelineResult], filepath: str):
        """Save pipeline results to JSON."""
        data = []
        for result in results:
            data.append({
                "query": result.query,
                "extracted_symptoms": result.extracted_symptoms,
                "kg_prediction": {
                    "disorder": result.kg_result["disorder"],
                    "confidence": result.kg_result["confidence"]
                },
                "llm_prediction": {
                    "disorder": result.llm_result["disorder"],
                    "confidence": result.llm_result["confidence"]
                },
                "fusion": result.fusion_result,
                "risk_level": result.risk_level.value,
                "should_abstain": result.should_abstain
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SEAL-KG: Hybrid KG + LLM System for Hallucination Reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run default pipeline
  python main.py --interactive            # Interactive mode
  python main.py --evaluate               # Run evaluation
  python main.py -i -e                     # Interactive + evaluation
  python main.py --query "I feel sad"     # Process single query
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for queries)"
    )
    
    parser.add_argument(
        "-e", "--evaluate",
        action="store_true",
        help="Run evaluation on sample queries"
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Process a single query"
    )
    
    parser.add_argument(
        "--use_qwen",
        action="store_true",
        help="Use actual Qwen model (requires GPU, more memory)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of sample queries to process (default: 3)"
    )
    
    return parser.parse_args()


def run_interactive(pipeline: HybridKGLLMPipeline):
    """Run in interactive mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'help' for available commands.")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n💬 Enter your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == "help":
                print("\n📖 Available commands:")
                print("  - Any mental health query")
                print("  - quit/exit: End session")
                print("  - help: Show this help")
                continue
            
            # Process query
            result = pipeline.process_query(query, verbose=True)
            print(f"\n{'='*70}")
            print("FINAL RESPONSE:")
            print(f"{'='*70}")
            print(result.final_response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("HYBRID KG + LLM (QWEN) SYSTEM FOR HALLUCINATION REDUCTION")
    print("=" * 70)
    print("\nThis system combines:")
    print("  - Knowledge Graph (KG) for symbolic reasoning")
    print("  - Qwen LLM for neural-based prediction")
    print("  - SEAL Module for safe abstention")
    print("  - Fusion Module for agreement detection")
    print("=" * 70)
    
    # Determine model to use
    model_name = "Qwen/Qwen2-0.5B" if args.use_qwen else "Qwen/Qwen2-0.5B"
    
    # Create pipeline
    pipeline = HybridKGLLMPipeline(
        model_name=model_name,
        agreement_threshold=0.5,
        confidence_threshold=0.6
    )
    
    # Handle different modes
    if args.query:
        # Single query mode
        result = pipeline.process_query(args.query, verbose=True)
        print(f"\n{'='*70}")
        print("FINAL RESPONSE:")
        print(f"{'='*70}")
        print(result.final_response)
        
    elif args.interactive:
        # Interactive mode
        run_interactive(pipeline)
        
    elif args.evaluate:
        # Evaluation mode
        pipeline.run_evaluation()
        
    else:
        # Default: process sample queries
        print("\n" + "=" * 70)
        print("PROCESSING SAMPLE QUERIES")
        print("=" * 70)
        
        for query in SAMPLE_QUERIES[:args.num_samples]:
            result = pipeline.process_query(query, verbose=True)
            print(f"\n{'='*70}")
            print("FINAL RESPONSE:")
            print(f"{'='*70}")
            print(result.final_response)
            print(f"\n{'='*70}\n")
        print(f"FINAL RESPONSE:")
        print(f"{'='*70}")
        print(result.final_response)
    
    # Run evaluation
    pipeline.run_evaluation()
    
    # Save results
    pipeline.save_results([], "results.json")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()