"""Quick system health check for all components."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def check_component(name, func):
    """Test a component and report status."""
    try:
        func()
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {str(e)[:50]}")
        return False


def check_data_pipeline():
    from src import data_pipeline
    training_df, weekly_df = data_pipeline.load_processed_datasets()
    assert not training_df.empty


def check_model_pipeline():
    from src import model_pipeline, data_pipeline
    X, _ = data_pipeline.prepare_training_data()
    sample = X.iloc[0].to_dict()
    result = model_pipeline.predict_single("random_forest", sample)
    assert "prediction" in result


def check_explainability():
    from src import explainability
    exp = explainability.build_explanation(
        "High", {"High": 0.8}, [0.5, -0.3], ["f1", "f2"]
    )
    assert exp.narrative is not None


def check_auditing():
    from src import auditing
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2]})
    report = auditing.log_data_quality(df)
    assert isinstance(report, pd.DataFrame)


def check_nlp():
    from src import nlp_layer
    top = nlp_layer.summarize_shap_values([0.5, -0.3], ["f1", "f2"])
    assert len(top) == 2


def check_api():
    from api.app import app
    assert app is not None


def check_files():
    required = [
        "data/processed/merged_training.csv",
        "models/random_forest.joblib",
        "config/model_config.yaml"
    ]
    for file in required:
        assert (BASE_DIR / file).exists(), f"Missing {file}"


def main():
    print("\n" + "=" * 60)
    print("SYSTEM HEALTH CHECK")
    print("=" * 60 + "\n")
    
    checks = [
        ("Required Files", check_files),
        ("Data Pipeline", check_data_pipeline),
        ("Model Pipeline", check_model_pipeline),
        ("Explainability", check_explainability),
        ("Auditing", check_auditing),
        ("NLP Layer", check_nlp),
        ("API", check_api),
    ]
    
    results = [check_component(name, func) for name, func in checks]
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULT: {passed}/{total} components healthy")
    print("=" * 60 + "\n")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
