#!/usr/bin/env python3
"""
Fixed script to run the explainability pipeline with correct paths.
This script ensures all paths point to the correct locations within the project.
"""

import os
from pathlib import Path
from ztiny import explain_and_evaluate

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

# Define correct paths
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_training.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "explanation_report.csv"

def main():
    """Run the explainability pipeline with correct paths."""
    
    # Verify all required files exist
    missing_files = []
    for name, path in [
        ("Model", MODEL_PATH),
        ("Vectorizer", VECTORIZER_PATH), 
        ("Test data", TEST_DATA_PATH)
    ]:
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return
    
    print("✅ All required files found. Starting explainability pipeline...")
    print(f"📊 Model: {MODEL_PATH}")
    print(f"🔤 Vectorizer: {VECTORIZER_PATH}")
    print(f"📄 Test data: {TEST_DATA_PATH}")
    print(f"📝 Output: {OUTPUT_PATH}")
    print("-" * 60)
    
    try:
        explanations = explain_and_evaluate(
            model_path=str(MODEL_PATH),
            vectorizer_path=str(VECTORIZER_PATH),
            test_path=str(TEST_DATA_PATH),
            text_column="contract_text",  # Use contract_text as the text column
            label_column="risk_label",    # Use risk_label as the label column
            output_path=str(OUTPUT_PATH),
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Generated {len(explanations)} explanations")
        print(f"📝 Results saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
