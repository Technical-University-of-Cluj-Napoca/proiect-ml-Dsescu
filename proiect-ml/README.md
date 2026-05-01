# Proiect Machine Learning — clasificare, regresie, SHAP și Streamlit

Acest proiect implementează complet cerințele pentru analiza comparată a modelelor de Machine Learning.

## Seturi de date

- `data/heart.csv` — clasificare, target: `HeartDisease`
- `data/insurance.csv` — regresie, target: `charges`

## Ce este implementat

Pentru fiecare problemă există câte un notebook complet:

- `notebooks/classification_heart_complete.ipynb`
- `notebooks/regression_insurance_complete.ipynb`

Notebook-urile includ:

1. definirea problemei;
2. EDA complet;
3. preprocesare cu `Pipeline` și `ColumnTransformer`;
4. split 75% train / 25% test;
5. antrenarea tuturor modelelor baseline;
6. ajustarea hiperparametrilor pentru primii 5 algoritmi;
7. curbe de învățare pentru primii 5 algoritmi;
8. SHAP pentru primii 5 algoritmi;
9. salvarea modelelor, metricilor și graficelor pentru Streamlit.

## Algoritmi incluși

### Clasificare

- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- XGBoost
- CatBoost
- Explainable Boosting Machine

### Regresie

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- K-Nearest Neighbor Regressor
- Gaussian Process Regressor
- XGBoost Regressor
- CatBoost Regressor
- Explainable Boosting Regressor

## Instalare cu uv

```bash
uv sync
```

## Rulare notebook-uri

```bash
jupyter notebook
```

Apoi deschide notebook-urile din folderul `notebooks/` și rulează celulele în ordine.

## Generare rapidă artefacte fără notebook

```bash
python src/train_all.py
```

Acest script creează folderul `outputs/`, unde vor fi salvate:

- metricile baseline;
- metricile după tuning;
- modelele `.joblib`;
- graficele EDA;
- curbele de învățare;
- graficele SHAP.

## Rulare Streamlit

După ce ai rulat notebook-urile sau `src/train_all.py`:

```bash
uv run streamlit run Home.py
```

În meniul din stânga apar paginile:

- `Clasificare Heart Disease`
- `Regresie Insurance`

## Observație despre timp de rulare

SHAP pentru toate modelele și tuningul Bayesian pot dura mai mult. Pentru rulare rapidă poți reduce în notebook:

```python
n_iter_bayes=6
sample_size=40
```

Pentru varianta finală a proiectului este mai bine să păstrezi valori mai mari.
