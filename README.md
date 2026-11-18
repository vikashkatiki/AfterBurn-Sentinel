# AfterBurn Sentinel 🔥

**Understanding the likelihood of debris flows after wildfires using machine learning.**

After a wildfire, heavy rain can wash soil and debris down hillsides. This can put nearby communities and roads at risk. AfterBurn Sentinel uses historical wildfire and rainfall records to study when these debris flows were observed.

The project brings together machine learning, maps and SHAP explanations in an interactive Streamlit dashboard.

## 🚀 Live Demo

**[Launch AfterBurn Sentinel – Live Application](https://afterburn-sentinal.streamlit.app/)**

**[View Source Code on GitHub](https://github.com/vikashkatiki/AfterBurn-Sentinel)**

Explore the dashboard to try historical likelihood assessment, view model explanations and examine geospatial prediction results.

> **Please note:** This is a research project based on historical records. It is not a live warning system and should not be used to make safety or evacuation decisions.

## Dataset

The project uses real data from the [U.S. Geological Survey (USGS)](https://doi.org/10.5066/P1R9VXC4), not a Kaggle dataset.

- 8,981 observations
- 80 wildfire areas across 8 U.S. states
- 1,140 observations where a debris flow was recorded

In the dataset, `Response = 1` means a debris flow was recorded. `Response = 0` means one was not recorded; it does not necessarily mean that no water or sediment moved.

## What the Project Does

1. Cleans the data and explores rainfall, soil, burn and watershed measurements.
2. Compares six machine-learning approaches.
3. Tests models while keeping observations from the same wildfire together.
4. Uses SHAP to explain which measurements contribute to model predictions.
5. Maps recorded observations and the model's correct and incorrect predictions.

The selected model is a Logistic Regression model using 10 measurements. The classification threshold is **0.47**.

## Model Results

The current evaluation produced these results:

| Measure | Result |
|---|---:|
| Accuracy | 86.97% |
| Precision | 35.31% |
| Recall | 76.61% |
| F1-score | 48.34% |
| ROC-AUC | 0.91 |
| PR-AUC | 0.6228 |

The model correctly identified 167 recorded debris-flow observations and missed 51. It also raised 306 false alarms.

**A note about these results:** The holdout data were not completely untouched during the earlier model-selection process. These numbers are useful for understanding the current model, but they should not be presented as a final unbiased test result. A stronger final evaluation would use a test set kept separate from the beginning or nested group-based validation.

## Dashboard Features

The Streamlit application includes four sections:

- **Project Overview:** A quick look at the dataset and response counts.
- **Risk Assessment:** Enter measurements and view the model's estimate.
- **Explainable AI:** See which measurements influenced the model overall and for an individual observation.
- **Geospatial Analysis:** Explore observation locations and where the model made errors.

## Run the Project Locally

Open a terminal in the project folder and install the required libraries:

```bash
python -m pip install streamlit streamlit-folium pandas numpy scikit-learn joblib shap folium matplotlib seaborn
```

Then start the dashboard:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

Keep the project's `data`, `models` and `outputs` folders in place. An internet connection is needed to load the map tiles.

## Data Source

Graber, A. P., et al. (2026). *USGS runoff-generated postfire debris-flow inventory*. U.S. Geological Survey data release.

[https://doi.org/10.5066/P1R9VXC4](https://doi.org/10.5066/P1R9VXC4)

## Project Repository

[https://github.com/vikashkatiki/AfterBurn-Sentinel](https://github.com/vikashkatiki/AfterBurn-Sentinel)