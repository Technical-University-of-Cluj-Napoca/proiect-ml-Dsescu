import pandas as pd
import streamlit as st
from pathlib import Path
from src.ml_utils import (
    CLASSIFICATION_TARGET, REGRESSION_TARGET, get_classification_scores,
    load_artifacts, load_dataset, make_local_shap_figures_for_input, rank_results, safe_name
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

def _task_settings(task: str) -> dict:
    if task == "classification":
        return {
            "title": "🩺 Predicție Boală Cardiacă (Clasificare)",
            "data_file": "heart.csv",
            "target": CLASSIFICATION_TARGET,
            "desc": "Acest model estimează riscul de boală cardiacă pe baza analizelor clinice."
        }
    return {
        "title": " Estimare Costuri Asigurare (Regresie)",
        "data_file": "insurance.csv",
        "target": REGRESSION_TARGET,
        "desc": "Acest model estimează costurile medicale anuale pe baza profilului pacientului."
    }

@st.cache_data(show_spinner=False)
def load_data(task: str):
    return load_dataset(DATA_DIR / _task_settings(task)["data_file"])

@st.cache_resource(show_spinner=False)
def cached_artifacts(task: str):
    return load_artifacts(task, OUTPUT_DIR)

def render_task_page(task: str):
    settings = _task_settings(task)
    df = load_data(task)
    target = settings["target"]

    st.title(settings["title"])
    st.markdown(f"**{settings['desc']}**")

    with st.expander("Explorare Dataset (EDA)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Linii", df.shape[0])
        c2.metric("Coloane", df.shape[1])
        c3.metric("Target", target)
        
        st.dataframe(df.head(5), use_container_width=True)
        
        task_dir = OUTPUT_DIR / task / "figures" / "eda"
        plots = ["01_target_distribution.png", "02_numeric_distributions.png", "04_correlation_matrix.png"]
        for p in plots:
            if (task_dir / p).exists():
                st.image(str(task_dir / p))

    try:
        artifacts = cached_artifacts(task)
    except Exception:
        st.error("Modelele nu au fost găsite. Rulează `python src/train_all.py` în terminal mai întâi.")
        return

    st.divider()
    model_name = st.selectbox("Selectează modelul optimizat pentru testare:", list(artifacts["models"].keys()))
    model = artifacts["models"][model_name]

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Performanța modelului:**")
        metrics_df = artifacts["tuned_metrics"]
        st.dataframe(metrics_df[metrics_df["Model"] == model_name].drop(columns=["Model"]), hide_index=True)
    with c2:
        with st.expander("Vezi hiperparametrii optimizați"):
            st.json(artifacts.get("best_params", {}).get(model_name, {}))

    with st.expander("Learning Curve (Overfitting Check)"):
        lc_path = OUTPUT_DIR / task / "figures" / "learning_curves" / f"learning_curve_{safe_name(model_name)}.png"
        if lc_path.exists():
            st.image(str(lc_path))
        else:
            st.info("Graficul nu este disponibil.")


    st.divider()
    st.subheader("Simulator Predicții")
    
    with st.form("prediction_form"):
        cols = st.columns(3)
        input_data = {}
        for i, col in enumerate([c for c in df.columns if c != target]):
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                val = cols[i % 3].number_input(col, value=float(series.mean()))
            else:
                val = cols[i % 3].selectbox(col, series.dropna().unique())
            input_data[col] = val
        
        submitted = st.form_submit_button("Analizează pacientul", type="primary")

    if submitted:
        X_input = pd.DataFrame([input_data])
        pred = model.predict(X_input)[0]
        
        st.markdown("### Rezultat")
        if task == "classification":
            if pred == 1:
                st.error("RISC RIDICAT DE BOALĂ CARDIACĂ")
            else:
                st.success("PACIENT SĂNĂTOS")
        else:
            st.info(f"Cost estimat: **${pred:,.2f}**")

        st.markdown("#### De ce a luat modelul această decizie? (SHAP Local)")
        try:
            figures = make_local_shap_figures_for_input(task, model, df.drop(columns=[target]), X_input)
            st.pyplot(figures["waterfall"])
        except Exception as e:
            st.warning("SHAP-ul local nu a putut fi generat pentru acest model.")
            
    with st.expander("Importanța globală a variabilelor (SHAP Summary)"):
        shap_path = OUTPUT_DIR / task / "figures" / "shap" / f"shap_summary_{safe_name(model_name)}.png"
        if shap_path.exists():
            st.image(str(shap_path))