# app.py — Interactive HTTP Dashboard for Prosopis juliflora Soil Nutrient ANN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Page configuration
st.set_page_config(page_title="Prosopis Soil Nutrient ANN Dashboard", layout="wide")
st.title("Prosopis juliflora Soil Nutrient Predictor (ANN Dashboard)")
st.markdown(
    "Interactive **Empirical ANN Model** predicting Available N, P, K, and Organic Carbon "
    "from *Prosopis* stand density, soil temperature, and microbial colony count."
)

# ------------------------------------------------------------------------------
# 1. CACHED DATA SIMULATION, MODEL TRAINING & 5-FOLD CV
# ------------------------------------------------------------------------------
@st.cache_resource
def train_all_models():
    np.random.seed(123)
    n_plots = 300

    density   = np.round(np.random.uniform(50, 1200, size=n_plots), 0)
    soil_temp = np.round(np.random.uniform(22, 40,   size=n_plots), 1)
    microbes  = np.round(np.random.uniform(50, 450,  size=n_plots), 0)
    temp_effect = np.exp(-((soil_temp - 30.0) ** 2) / (2 * 5.0 ** 2))

    soil_data = pd.DataFrame({
        "Plot_ID":          [f"Plot_{i}" for i in range(1, n_plots + 1)],
        "Prosopis_Density": density,
        "Soil_Temp_C":      soil_temp,
        "Microbial_CFU":    microbes,
        "Nitrogen_mg_kg":   75   + (0.045  * density + 0.09  * microbes) * (0.5 + 0.8 * temp_effect) + np.random.normal(0, 5.0,  size=n_plots),
        "Phosphorus_mg_kg": 3.5  + (0.0038 * density + 0.007 * microbes) * (0.5 + 0.8 * temp_effect) + np.random.normal(0, 0.45, size=n_plots),
        "Potassium_mg_kg":  80   + (0.030  * density + 0.06  * microbes) * (0.5 + 0.7 * temp_effect) + np.random.normal(0, 4.0,  size=n_plots),
        "OC_percent":       0.28 + (0.00025 * density + 0.0004 * microbes) * (0.5 + 0.8 * temp_effect) + np.random.normal(0, 0.03, size=n_plots),
    })

    features = ["Prosopis_Density", "Soil_Temp_C", "Microbial_CFU"]
    nutrients = {
        "Nitrogen_mg_kg":   "Available Nitrogen (mg/kg)",
        "Phosphorus_mg_kg": "Available Phosphorus (mg/kg)",
        "Potassium_mg_kg":  "Available Potassium (mg/kg)",
        "OC_percent":       "Soil Organic Carbon (%)",
    }

    train_idx, test_idx = train_test_split(soil_data.index, test_size=0.20, random_state=123)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(soil_data.loc[train_idx, features])
    X_test_scaled  = scaler.transform(soil_data.loc[test_idx, features])

    ann_models, test_results, comp_rows, cv_rows = {}, {}, [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=123)

    for col, label in nutrients.items():
        y_train = soil_data.loc[train_idx, col]
        y_test  = soil_data.loc[test_idx,  col]

        ann = MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs",
            alpha=0.01, max_iter=1000, random_state=123
        )
        ann.fit(X_train_scaled, y_train)
        y_pred_ann = ann.predict(X_test_scaled)

        mlr = LinearRegression().fit(X_train_scaled, y_train)
        y_pred_mlr = mlr.predict(X_test_scaled)

        ann_models[col] = ann
        test_results[col] = pd.DataFrame({
            "Actual": y_test.values,
            "ANN_Predicted": y_pred_ann,
            "MLR_Predicted": y_pred_mlr
        })

        comp_rows.append({
            "Nutrient": label,
            "MLR R²":   round(r2_score(y_test, y_pred_mlr), 3),
            "ANN R²":   round(r2_score(y_test, y_pred_ann), 3),
            "MLR RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_mlr)), 3),
            "ANN RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_ann)), 3),
        })

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ann", MLPRegressor(
                hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs",
                alpha=0.01, max_iter=1000, random_state=123
            ))
        ])
        scores = cross_validate(
            pipe, soil_data[features], soil_data[col], cv=kf,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error")
        )
        bias = np.mean(y_pred_ann - y_test.values)
        cv_rows.append({
            "Nutrient": label,
            "CV R² (Mean ± SD)": f"{scores['test_r2'].mean():.3f} ± {scores['test_r2'].std():.3f}",
            "CV RMSE": round(-scores["test_neg_root_mean_squared_error"].mean(), 3),
            "CV MAE":  round(-scores["test_neg_mean_absolute_error"].mean(), 3),
            "Test Mean Bias": round(bias, 3),
        })

    return soil_data, features, nutrients, scaler, ann_models, test_results, pd.DataFrame(comp_rows), pd.DataFrame(cv_rows)

soil_data, features, nutrients, scaler, ann_models, test_results, comp_df, cv_df = train_all_models()

# ------------------------------------------------------------------------------
# 2. SIDEBAR: LIVE INTERACTIVE INPUTS
# ------------------------------------------------------------------------------
st.sidebar.header("Scenario Controls")
density_in  = st.sidebar.slider("Prosopis Density (trees/ha)", 0, 1500, 750, step=50)
temp_in     = st.sidebar.slider("Soil Temperature (°C)", 20.0, 42.0, 30.0, step=0.5)
microbes_in = st.sidebar.slider("Microbial Count (× 10⁵ CFU/g)", 50, 450, 250, step=10)

if density_in > 1200 or density_in < 50:
    st.sidebar.warning("Extrapolation zone: Density is outside the 50–1200 training range.")

# Live Point-Estimate Prediction
input_df = pd.DataFrame({
    "Prosopis_Density": [density_in],
    "Soil_Temp_C":      [temp_in],
    "Microbial_CFU":    [microbes_in],
})
input_scaled = scaler.transform(input_df)

pred_N  = ann_models["Nitrogen_mg_kg"].predict(input_scaled)[0]
pred_P  = ann_models["Phosphorus_mg_kg"].predict(input_scaled)[0]
pred_K  = ann_models["Potassium_mg_kg"].predict(input_scaled)[0]
pred_OC = ann_models["OC_percent"].predict(input_scaled)[0]

# ------------------------------------------------------------------------------
# 3. LIVE KPI METRIC CARDS
# ------------------------------------------------------------------------------
st.subheader("Live ANN Point-Estimate Predictions")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Available Nitrogen",   f"{pred_N:.2f} mg/kg")
c2.metric("Available Phosphorus", f"{pred_P:.2f} mg/kg")
c3.metric("Available Potassium",  f"{pred_K:.2f} mg/kg")
c4.metric("Soil Organic Carbon",  f"{pred_OC:.3f} %")

# ------------------------------------------------------------------------------
# 4. DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Response Curves", "Model Comparison & Validation", "Residual Diagnostics"])

with tab1:
    st.markdown(f"**ANN Response across Prosopis Density** (at Temp = `{temp_in}°C`, Microbes = `{microbes_in} × 10⁵ CFU/g`)")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    x_grid = pd.DataFrame({
        "Prosopis_Density": np.linspace(0, 1500, 150),
        "Soil_Temp_C":      temp_in,
        "Microbial_CFU":    microbes_in,
    })
    X_grid_scaled = scaler.transform(x_grid[features])

    for ax, (col, label), current_val in zip(axes.flatten(), nutrients.items(), [pred_N, pred_P, pred_K, pred_OC]):
        ax.scatter(soil_data["Prosopis_Density"], soil_data[col], alpha=0.2, color="#7F8C8D", s=18)
        ax.plot(x_grid["Prosopis_Density"], ann_models[col].predict(X_grid_scaled), color="#1B7A43", linewidth=2)
        ax.scatter([density_in], [current_val], color="#C0392B", s=90, zorder=5)
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xlabel("Prosopis Density (trees/ha)")
        ax.set_ylabel("Concentration")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("ANN vs. Multiple Linear Regression (20% Holdout Test Set)")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("5-Fold Cross-Validation & Bias Summary")
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("**Observed − Predicted Residuals (Test Set)**")
    fig_res, axes_res = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (col, label) in zip(axes_res.flatten(), nutrients.items()):
        df_r = test_results[col]
        residuals = df_r["Actual"] - df_r["ANN_Predicted"]
        mae_val = mean_absolute_error(df_r["Actual"], df_r["ANN_Predicted"])
        ax.scatter(df_r["ANN_Predicted"], residuals, alpha=0.75, color="#2980B9", edgecolor="white", s=40)
        ax.axhline(0, color="#C0392B", linestyle="--", linewidth=1.5)
        ax.set_title(f"{label} (MAE = {mae_val:.2f})", fontweight="bold", fontsize=10)
        ax.set_xlabel("ANN Predicted")
        ax.set_ylabel("Residual")
    plt.tight_layout()
    st.pyplot(fig_res)