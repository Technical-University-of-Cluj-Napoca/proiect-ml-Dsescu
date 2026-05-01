from pathlib import Path
from ml_utils import (
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    execute_complete_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    print("=== CLASIFICARE: Heart Disease ===")
    execute_complete_pipeline(
        task="classification",
        csv_path=DATA_DIR / "heart.csv",
        target=CLASSIFICATION_TARGET,
        output_root=OUTPUT_DIR,
        top_n=5,
        cv=3,
        n_iter_bayes=12,
        shap_sample_size=80,
    )

    print("=== REGRESIE: Medical Insurance Charges ===")
    execute_complete_pipeline(
        task="regression",
        csv_path=DATA_DIR / "insurance.csv",
        target=REGRESSION_TARGET,
        output_root=OUTPUT_DIR,
        top_n=5,
        cv=3,
        n_iter_bayes=12,
        shap_sample_size=80,
    )

    print("Gata. Artefactele au fost salvate in folderul outputs/.")


if __name__ == "__main__":
    main()
