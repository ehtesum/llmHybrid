# =============================================================================
# SEAL-KG: Google Colab Quick Start
# =============================================================================
# 
# Colab Setup:
# 1. Open: https://colab.research.google.com
# 2. New Notebook → Upload files OR clone repo
# 3. Runtime → Change runtime type → GPU (recommended for Qwen)
# 4. Run cells in order
#
# Quick Commands:
#   !python main.py --evaluate    # Run evaluation
#   !python train.py --epochs 3   # Run training
#   !python main.py -i            # Interactive mode
# =============================================================================

# @title 📦 Step 1: Clone Repository & Setup
# If using local files, skip this and upload files directly
!git clone https://github.com/your-repo/seal-kg.git seal-kg 2>/dev/null || echo "Using local files..."
%cd seal-kg 2>/dev/null || %cd .

# @title 📦 Step 2: Install Dependencies
print("Installing dependencies...")
!pip install -q transformers torch tqdm sentence-transformers
print("✓ Dependencies installed")

# @title 📦 Step 3: Import Modules
import sys
sys.path.append('.')
print("Importing modules...")

from kg import KnowledgeGraph
from symptom_extraction import SymptomExtractor
from llm_qwen import QwenLLM
from fusion import FusionModule
from seal import SealModule, RiskLevel
from evaluation import create_evaluator

print("✓ All modules imported")

# @title 🔧 Initialize Pipeline
print("\n" + "="*50)
print("Initializing SEAL-KG Pipeline...")
print("="*50)

kg = KnowledgeGraph()
extractor = SymptomExtractor()
llm = QwenLLM()
fusion = FusionModule()
seal = SealModule()
evaluator = create_evaluator()

print("✓ Pipeline initialized")

# @title 🔍 Test: Run Sample Query
query = "I feel sad and lost interest in everything"

print(f"\n📝 Query: {query}")
print("-"*50)

# Stage 1: Symptom Extraction
symptoms = extractor.extract_symptoms(query)
print(f"1️⃣ Symptoms: {symptoms}")

# Stage 2a: KG Query
kg_results = kg.query(symptoms)
print(f"2️⃣ KG: {kg_results[0] if kg_results else 'Unknown'}")

# Stage 2b: LLM Prediction
llm_disorder, llm_conf, _ = llm.predict(symptoms)
print(f"3️⃣ LLM: {llm_disorder} ({llm_conf:.2f})")

# Stage 3: Fusion
kg_result = {"disorder": kg_results[0][0], "confidence": kg_results[0][1]} if kg_results else {"disorder": "Unknown", "confidence": 0.0}
llm_result = {"disorder": llm_disorder, "confidence": llm_conf}
fusion_result = fusion.fuse(kg_result, llm_result)

print(f"4️⃣ Agreement: {fusion_result['agreement_score']:.2f}")
print(f"5️⃣ Decision: {fusion_result['decision']}")

# Risk Classification
risk_level, _ = seal.classify_risk(query)
print(f"6️⃣ Risk: {risk_level.value.upper()}")

print("\n" + "="*50)
print("✅ Test complete! System working.")
print("="*50)

# @title 🚀 Run Full Pipeline
# Uncomment to run:
# !python main.py

# @title 📊 Run Training
# Uncomment to run:
# !python train.py --epochs 3

# @title 📈 Run Evaluation
# Uncomment to run:
# !python main.py --evaluate