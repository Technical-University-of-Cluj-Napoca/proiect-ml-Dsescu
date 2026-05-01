import streamlit as st

st.set_page_config(
    page_title="ML Comparative Analysis",
    page_icon="🤖",
    layout="wide",
)

st.title("Analiza comparată a modelelor de Machine Learning")
st.write(
    "Aplicația are două pagini în meniul din stânga: clasificare pentru Heart Disease "
    "și regresie pentru Insurance Charges."
)

st.markdown(
    """
### Pași de rulare

```bash
pip install -r requirements.txt
python src/train_all.py
streamlit run app.py
```

Dacă folosești `uv`:

```bash
uv sync
uv run python src/train_all.py
uv run streamlit run app.py
```
"""
)
