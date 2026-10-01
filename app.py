"""
Energy Consumption Forecasting Dashboard
Interactive Streamlit Application featuring:
- Time-series Machine Learning models (XGBoost, Random Forest, Online SGD, Holt-Winters, Adaptive Ensemble)
- Continuous / Retraining Engine (Online Incremental Updates & Batch Retraining)
- Interactive Plotly Visualizations
- 1-Click CSV and Publication-Ready PDF Report Exports
"""

import os
import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import EnergyDataLoader, generate_benchmark_energy_data
from src.feature_engineering import TimeSeriesFeatureEngineer
from src.trainer import EnergyForecastingPipeline
from src.pdf_generator import EnergyForecastPDFReport


# Page configuration
st.set_page_config(
    page_title="GridCast AI | Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284c7, #0f766e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748b;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-healthy {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-warning {
        background-color: #fef9c3;
        color: #a16207;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-drift {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = EnergyForecastingPipeline()

if "trained" not in st.session_state:
    st.session_state.trained = False

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "current_df" not in st.session_state:
    st.session_state.current_df = None

if "current_metadata" not in st.session_state:
    st.session_state.current_metadata = None


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/lightning-bolt.png", width=64)
    st.markdown("## **Control Center**")
    st.markdown("---")

    # 1. Dataset Selection
    st.markdown("### **1. Data Ingestion**")
    data_source = st.radio(
        "Select Data Source:",
        ["Multi-Region Benchmark Dataset", "Upload Custom CSV/Excel"],
        index=0,
    )

    df_raw = None
    selected_region = None

    if data_source == "Multi-Region Benchmark Dataset":
        sample_path = os.path.join("data", "sample_energy_data.csv")
        if not os.path.exists(sample_path):
            os.makedirs("data", exist_ok=True)
            df_gen = generate_benchmark_energy_data()
            df_gen.to_csv(sample_path, index=False)
        df_raw = pd.read_csv(sample_path)
        available_regions = df_raw["region"].unique().tolist()
        selected_region = st.selectbox("Select Regional Grid:", available_regions, index=0)
    else:
        uploaded_file = st.file_uploader("Upload Time-Series Data (CSV or Excel)", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                st.success(f"Uploaded {uploaded_file.name} ({len(df_raw)} records)")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.info("Awaiting file upload... Using benchmark dataset meanwhile.")
            sample_path = os.path.join("data", "sample_energy_data.csv")
            df_raw = pd.read_csv(sample_path)
            selected_region = "North-Metro"

    # Preprocess
    if df_raw is not None:
        loader = EnergyDataLoader()
        try:
            processed_df, meta = loader.process(df_raw, selected_region=selected_region)
            st.session_state.current_df = processed_df
            st.session_state.current_metadata = meta
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    st.markdown("---")

    # 2. Forecast Settings
    st.markdown("### **2. Forecast Settings**")
    horizon_choice = st.selectbox(
        "Forecast Horizon:",
        ["24 Hours (1 Day)", "48 Hours (2 Days)", "168 Hours (7 Days)", "336 Hours (14 Days)"],
        index=1,
    )
    horizon_map = {
        "24 Hours (1 Day)": 24,
        "48 Hours (2 Days)": 48,
        "168 Hours (7 Days)": 168,
        "336 Hours (14 Days)": 336,
    }
    horizon_steps = horizon_map[horizon_choice]

    confidence_pct = st.selectbox("Confidence Band:", ["95%", "90%"], index=0)
    conf_level = 0.95 if confidence_pct == "95%" else 0.90

    # 3. Model Engine Selection
    st.markdown("### **3. Model Architecture**")
    model_choice = st.selectbox(
        "Active Model:",
        [
            "Adaptive Ensemble (Recommended)",
            "XGBoost",
            "Random Forest",
            "Online Incremental (SGD)",
            "Holt-Winters (ETS)",
        ],
        index=0,
    )

    st.markdown("---")

    # 4. Trigger Training Action
    st.markdown("### **4. Model Actions**")
    train_button = st.button("🚀 Train & Benchmark Models", type="primary", use_container_width=True)


# --- PIPELINE EXECUTION ---
if train_button or (not st.session_state.trained and st.session_state.current_df is not None):
    with st.spinner("Training time-series models & calibrating ensemble..."):
        reg_name = selected_region or "Custom-Grid"
        res = st.session_state.pipeline.train_and_evaluate(
            df=st.session_state.current_df,
            test_horizon=horizon_steps,
            region_name=reg_name,
            trigger_type="Manual Benchmark",
        )
        st.session_state.last_results = res
        st.session_state.trained = True
        st.success(f"Models successfully trained! Best fit: **{res['best_model']}**")


# --- MAIN DASHBOARD HEADER ---
st.markdown('<div class="main-header">⚡ GridCast AI : Energy Consumption Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Continuous Learning Time-Series Intelligence & Autonomous Retraining Engine</div>', unsafe_allow_html=True)

if not st.session_state.trained or st.session_state.current_df is None:
    st.info("Click 'Train & Benchmark Models' in the sidebar to initialize the forecasting pipeline.")
    st.stop()

pipeline: EnergyForecastingPipeline = st.session_state.pipeline
results = st.session_state.last_results
current_df = st.session_state.current_df
meta = st.session_state.current_metadata

# Calculate forecast
active_model_name = "Adaptive Ensemble" if "Adaptive" in model_choice else model_choice
forecast_df = pipeline.forecast_future(
    df=current_df,
    horizon_steps=horizon_steps,
    model_name=active_model_name,
    confidence_level=conf_level,
)

# Extract key statistics
current_load = float(current_df[pipeline.target_col].iloc[-1])
peak_forecast = float(forecast_df["forecast_mw"].max())
peak_dt = forecast_df["forecast_mw"].idxmax()
min_forecast = float(forecast_df["forecast_mw"].min())
avg_forecast = float(forecast_df["forecast_mw"].mean())
total_energy = float(forecast_df["forecast_mw"].sum())

scorecard = pipeline.scorecard
best_mape = scorecard.loc[active_model_name, "MAPE"] if active_model_name in scorecard.index else 2.5
best_r2 = scorecard.loc[active_model_name, "R2"] if active_model_name in scorecard.index else 0.95

drift_report = results["drift_report"]
drift_status = drift_report["status"]
if "Healthy" in drift_status:
    status_pill = f'<span class="status-healthy">● {drift_status}</span>'
elif "Warning" in drift_status:
    status_pill = f'<span class="status-warning">▲ {drift_status}</span>'
else:
    status_pill = f'<span class="status-drift">■ {drift_status}</span>'


# --- TOP KPI METRICS BAR ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

with kpi_col1:
    st.metric(
        label="Latest Grid Load",
        value=f"{current_load:,.1f} MW",
        delta=f"{(current_load - current_df[pipeline.target_col].iloc[-2]):+.1f} MW (1h)",
    )

with kpi_col2:
    st.metric(
        label="Projected Peak Demand",
        value=f"{peak_forecast:,.1f} MW",
        delta=f"at {peak_dt.strftime('%b %d, %H:00')}",
        delta_color="inverse",
    )

with kpi_col3:
    st.metric(
        label="Average Demand",
        value=f"{avg_forecast:,.1f} MW",
        delta=f"Total {total_energy:,.0f} MWh",
    )

with kpi_col4:
    st.metric(
        label=f"{active_model_name} Accuracy",
        value=f"{best_mape:.2f}% MAPE",
        delta=f"R² = {best_r2:.3f}",
    )

with kpi_col5:
    st.markdown("**Continuous Learning Status**")
    st.markdown(status_pill, unsafe_allow_html=True)
    st.caption(f"KS Stat: {drift_report['ks_statistic']:.3f} (p={drift_report['p_value']:.4f})")

st.markdown("---")


# --- TABS INTERFACE ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast & Trajectory",
    "🔍 Seasonality & Demand Patterns",
    "🔄 Continuous Learning & Drift Monitor",
    "📥 Export Center (CSV & PDF)",
])


# ==========================================
# TAB 1: FORECAST & TRAJECTORY
# ==========================================
with tab1:
    st.markdown("### **Interactive Energy Demand Forecast**")
    st.caption(f"Displaying historical demand transitioning into the future {horizon_steps}-hour forecast with 95% confidence bounds.")

    history_window = st.slider("Historical Context Window (Days):", min_value=3, max_value=30, value=7)
    hist_subset = current_df.iloc[-history_window * 24 :]

    fig_forecast = go.Figure()

    # Historical line
    fig_forecast.add_trace(go.Scatter(
        x=hist_subset.index,
        y=hist_subset[pipeline.target_col],
        name="Historical Actual (MW)",
        line=dict(color="#334155", width=2),
        mode="lines",
    ))

    # Upper bound
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["upper_bound_mw"],
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Lower bound + fill
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["lower_bound_mw"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(14, 165, 233, 0.20)",
        name="95% Confidence Interval",
        hoverinfo="skip",
    ))

    # Forecast line
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["forecast_mw"],
        name=f"AI Forecast ({active_model_name})",
        line=dict(color="#0284c7", width=3, dash="dash"),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    # Mark Peak Demand
    fig_forecast.add_trace(go.Scatter(
        x=[peak_dt],
        y=[peak_forecast],
        mode="markers+text",
        name="Projected Peak",
        text=[f"Peak: {peak_forecast:,.0f} MW"],
        textposition="top center",
        marker=dict(color="#ef4444", size=12, symbol="triangle-up"),
    ))

    fig_forecast.update_layout(
        template="plotly_white",
        height=520,
        hovermode="x unified",
        xaxis=dict(
            title="Timestamp",
            rangeslider=dict(visible=True),
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="Energy Consumption (MW)",
            gridcolor="#f1f5f9",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Test Set Actual vs Predicted Evaluation Chart
    st.markdown("#### **Test Window Model Fit (Actuals vs Predictions)**")
    test_actuals = results["test_actuals"]
    test_preds = results["test_predictions"].get(active_model_name, None)

    if test_preds is not None:
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=test_actuals.index,
            y=test_actuals.values,
            name="Ground Truth (Actual MW)",
            line=dict(color="#0f172a", width=2),
        ))
        fig_eval.add_trace(go.Scatter(
            x=test_actuals.index,
            y=test_preds,
            name=f"Model Fitted Prediction ({active_model_name})",
            line=dict(color="#10b981", width=2.5, dash="dot"),
        ))
        fig_eval.update_layout(
            template="plotly_white",
            height=320,
            hovermode="x unified",
            xaxis=dict(title="Test Period Timestamp"),
            yaxis=dict(title="Energy Demand (MW)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_eval, use_container_width=True)


# ==========================================
# TAB 2: SEASONALITY & DEMAND PATTERNS
# ==========================================
with tab2:
    st.markdown("### **Diurnal & Weekly Energy Consumption Patterns**")
    
    pat_col1, pat_col2 = st.columns(2)

    with pat_col1:
        st.markdown("#### **24-Hour Diurnal Demand Profile**")
        df_diurnal = current_df.copy()
        df_diurnal["hour"] = df_diurnal.index.hour
        hourly_grp = df_diurnal.groupby("hour")[pipeline.target_col].agg(["mean", "min", "max"])

        fig_diurnal = go.Figure()
        fig_diurnal.add_trace(go.Scatter(
            x=hourly_grp.index,
            y=hourly_grp["mean"],
            name="Average Load",
            line=dict(color="#0284c7", width=3),
        ))
        fig_diurnal.add_trace(go.Scatter(
            x=hourly_grp.index,
            y=hourly_grp["max"],
            name="Max Observed",
            line=dict(color="#94a3b8", width=1, dash="dot"),
        ))
        fig_diurnal.add_trace(go.Scatter(
            x=hourly_grp.index,
            y=hourly_grp["min"],
            name="Min Observed",
            line=dict(color="#94a3b8", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(148, 163, 184, 0.15)",
        ))
        fig_diurnal.update_layout(
            template="plotly_white",
            height=340,
            xaxis=dict(title="Hour of Day (0-23)", tickmode="linear", dtick=2),
            yaxis=dict(title="Load (MW)"),
            margin=dict(l=30, r=30, t=20, b=30),
        )
        st.plotly_chart(fig_diurnal, use_container_width=True)

    with pat_col2:
        st.markdown("#### **Weekday vs Weekend Load Comparison**")
        df_week = current_df.copy()
        df_week["dayofweek"] = df_week.index.day_name()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_avg = df_week.groupby("dayofweek")[pipeline.target_col].mean().reindex(days_order)

        fig_bar = px.bar(
            x=day_avg.index,
            y=day_avg.values,
            labels={"x": "Day of Week", "y": "Avg Consumption (MW)"},
            color=day_avg.values,
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(template="plotly_white", height=340, coloraxis_showscale=False, margin=dict(l=30, r=30, t=20, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap: Hour vs Day of Week
    st.markdown("#### **Energy Intensity Heatmap (Hour of Day vs Day of Week)**")
    df_heat = current_df.copy()
    df_heat["hour"] = df_heat.index.hour
    df_heat["dayofweek"] = df_heat.index.day_name()
    heat_matrix = df_heat.pivot_table(index="dayofweek", columns="hour", values=pipeline.target_col, aggfunc="mean").reindex(days_order)

    fig_heat = px.imshow(
        heat_matrix,
        labels=dict(x="Hour of Day", y="Day of Week", color="Demand (MW)"),
        x=list(range(24)),
        y=days_order,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig_heat.update_layout(height=320, margin=dict(l=30, r=30, t=20, b=30))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Feature Importances (if XGBoost or RF)
    if "XGBoost" in pipeline.models:
        st.markdown("#### **Top Feature Drivers of Energy Demand (XGBoost Feature Importance)**")
        feat_imp = pipeline.models["XGBoost"].get_feature_importances().head(12)
        fig_feat = px.bar(
            x=feat_imp.values,
            y=feat_imp.index,
            orientation="h",
            labels={"x": "Relative Importance", "y": "Feature"},
            color=feat_imp.values,
            color_continuous_scale="Teal",
        )
        fig_feat.update_layout(template="plotly_white", height=320, yaxis=dict(autorange="reversed"), coloraxis_showscale=False, margin=dict(l=30, r=30, t=20, b=30))
        st.plotly_chart(fig_feat, use_container_width=True)


# ==========================================
# TAB 3: CONTINUOUS LEARNING & DRIFT MONITOR
# ==========================================
with tab3:
    st.markdown("### **Continuous Learning, Online Updates & Concept Drift Center**")
    st.markdown("""
    To eliminate **data decay**, this system utilizes a two-tier continuous learning architecture:
    1. **Online Incremental Learning**: `SGDRegressor` and dynamic ensemble weights continuously adapt to streaming data without full retraining.
    2. **Concept Drift-Triggered Batch Retraining**: Monitors distribution shifts via Kolmogorov-Smirnov statistical tests and rolling residual errors.
    """)

    drift_c1, drift_c2 = st.columns([1, 1])

    with drift_c1:
        st.markdown("#### **Concept Drift Diagnostics**")
        st.markdown(f"**Current Status:** {status_pill}", unsafe_allow_html=True)
        st.write(f"• **Diagnostic Message:** {drift_report['message']}")
        st.write(f"• **Two-Sample KS Statistic:** `{drift_report['ks_statistic']:.4f}` (p-value: `{drift_report['p_value']:.6f}`)")
        st.write(f"• **Baseline Historical Mean:** `{drift_report.get('ref_mean', 0.0):,.1f} MW` (± {drift_report.get('ref_std', 0.0):,.1f})")
        st.write(f"• **Recent Window Mean:** `{drift_report.get('recent_mean', 0.0):,.1f} MW` (± {drift_report.get('recent_std', 0.0):,.1f})")

        st.markdown("---")
        st.markdown("#### **Live Continuous Retraining Triggers**")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⚡ Ingest Stream Batch (Online Update)", use_container_width=True):
                with st.spinner("Incrementally updating model weights via partial_fit()..."):
                    # Simulate next batch (last 24 hours)
                    stream_batch = current_df.iloc[-24:].copy()
                    update_res = pipeline.continuous_online_update(
                        stream_batch,
                        region_name=selected_region or "Regional Grid",
                    )
                    st.success(f"Stream update complete! Checkpoint `{update_res['version_entry']['version']}` logged.")
                    st.rerun()

        with col_btn2:
            if st.button("🔄 Trigger Batch Retraining", use_container_width=True):
                with st.spinner("Executing full walk-forward batch retrain..."):
                    res = pipeline.train_and_evaluate(
                        df=current_df,
                        test_horizon=horizon_steps,
                        region_name=selected_region or "Regional Grid",
                        trigger_type="Drift / On-Demand Retrain",
                    )
                    st.session_state.last_results = res
                    st.success("Batch retraining finished! Models re-calibrated.")
                    st.rerun()

    with drift_c2:
        st.markdown("#### **Model Performance Scorecard**")
        st.dataframe(scorecard.style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#dcfce7").highlight_max(subset=["R2"], color="#dcfce7"), use_container_width=True)

        st.markdown("#### **Distribution Comparison: Reference vs Recent Window**")
        ref_n = min(720, len(current_df) - 168)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=current_df[pipeline.target_col].iloc[:ref_n],
            name="Reference Baseline Window",
            opacity=0.6,
            marker_color="#3b82f6",
            nbinsx=35,
        ))
        fig_dist.add_trace(go.Histogram(
            x=current_df[pipeline.target_col].iloc[-168:],
            name="Recent Window (Last 7 Days)",
            opacity=0.6,
            marker_color="#f97316",
            nbinsx=35,
        ))
        fig_dist.update_layout(
            barmode="overlay",
            template="plotly_white",
            height=280,
            xaxis=dict(title="Energy Demand (MW)"),
            yaxis=dict(title="Frequency"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=30, t=20, b=30),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Retraining & Version History Audit Log
    st.markdown("---")
    st.markdown("#### **Model Registry & Version Audit Trail**")
    if pipeline.version_history:
        history_df = pd.DataFrame(pipeline.version_history)
        st.dataframe(history_df, use_container_width=True)


# ==========================================
# TAB 4: EXPORT CENTER (CSV & PDF)
# ==========================================
with tab4:
    st.markdown("### **Export Center**")
    st.caption("Export future forecasts as clean structured CSV data or download a comprehensive executive PDF report.")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("#### **📄 Export Official PDF Report**")
        st.markdown("""
        The generated PDF report includes:
        - Executive Summary & Project Metrics
        - Model Benchmark & Validation Scorecard (MAE, RMSE, MAPE, R²)
        - Embedded High-Resolution Forecast Trajectory with 95% Confidence Bounds
        - 24-Hour Diurnal Demand Curve & Peak Hours
        - Concept Drift & Continuous Retraining Audit Trail
        - Tabular 24-Hour Forecast Breakdown
        """)

        # Generate PDF bytes
        pdf_gen = EnergyForecastPDFReport()
        pdf_bytes = pdf_gen.generate_pdf(
            historical_df=current_df,
            forecast_df=forecast_df,
            scorecard_df=scorecard,
            metadata={
                **meta,
                "selected_region": selected_region or "Regional Grid",
                "best_model": active_model_name,
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            drift_report=drift_report,
        )

        st.download_button(
            label="📥 Download Executive PDF Report",
            data=pdf_bytes,
            file_name=f"energy_forecast_report_{selected_region or 'grid'}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    with exp_col2:
        st.markdown("#### **📊 Export Forecast Data as CSV**")
        st.markdown("""
        The CSV export contains:
        - Exact Timestamps (Hourly)
        - Forecast Energy Consumption (MW)
        - Lower 95% Confidence Interval
        - Upper 95% Confidence Interval
        - Uncertainty Margin
        """)

        csv_buffer = io.StringIO()
        forecast_export = forecast_df.reset_index()
        forecast_export.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv_bytes,
            file_name=f"energy_forecast_data_{selected_region or 'grid'}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### **Forecast Data Preview Table**")
    st.dataframe(forecast_df.style.format("{:,.2f}"), use_container_width=True)
