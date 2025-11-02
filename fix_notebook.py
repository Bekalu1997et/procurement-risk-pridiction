import json

# Read notebook
with open('notebooks/07_complete_mlops_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 10 (Model Training)
nb['cells'][10]['source'] = [
    "print(\"\\n🤖 STEP 4: Model Training\")\\n",
    "print(\"=\" * 70)\\n",
    "\\n",
    "# Train models\\n",
    "artifacts = model_pipeline.train_models(X, y)\\n",
    "\\n",
    "print(\"\\nModel Training Complete:\")\\n",
    "for artifact in artifacts:\\n",
    "    print(f\"\\n{artifact.model_name}:\")\\n",
    "    print(f\"  Macro F1: {artifact.report['macro avg']['f1-score']:.4f}\")\\n",
    "    print(f\"  Accuracy: {artifact.report['accuracy']:.4f}\")\\n",
    "    print(f\"  Model saved: models/{artifact.model_name}.joblib\")\\n",
    "\\n",
    "print(\"\\n✓ Models trained and persisted\")"
]

# Clear outputs
nb['cells'][10]['outputs'] = []
nb['cells'][10]['execution_count'] = None

# Write back
with open('notebooks/07_complete_mlops_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook fixed successfully!")
