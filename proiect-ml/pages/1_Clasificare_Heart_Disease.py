import streamlit as st
from src.streamlit_page import render_task_page

st.set_page_config(page_title="Clasificare Heart Disease", layout="wide")
render_task_page("classification")
