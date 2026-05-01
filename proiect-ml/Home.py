import streamlit as st

st.set_page_config(page_title="ML Compare", page_icon="🤖", layout="wide")

st.title("ML Model Comparer")
st.markdown("### Analiză Comparativă: Clasificare vs. Regresie")

st.info("Folosește meniul din stânga pentru a naviga între problemele rezolvate.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Clasificare
    **Dataset:** Heart Disease  
    **Scop:** Prezicerea prezenței bolilor cardiace (1) sau absenței lor (0).  
    **Metrici principale:** F1-Score, ROC-AUC, Accuracy.
    """)

with col2:
    st.markdown("""
    ### Regresie
    **Dataset:** Medical Insurance  
    **Scop:** Estimarea costului anual cu asigurarea medicală pe baza profilului pacientului.  
    **Metrici principale:** RMSE, MAE, R2 Score.
    """)

st.divider()

st.markdown("""
###  Funcționalități aplicație:
* **Performanță Comparată:** Căutare Bayesiană / GridSearch pe Top 5 algoritmi.
* **Evaluare:** Curbe de învățare pentru detectarea de overfitting.
* **Explicabilitate XAI:** Evaluarea impactului decizional folosind SHAP (Global & Local).
""")