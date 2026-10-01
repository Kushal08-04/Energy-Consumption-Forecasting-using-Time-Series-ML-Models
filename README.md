# ⚡ GridCast AI: Energy Consumption Forecasting

An end-to-end, production-grade Machine Learning system and modern web dashboard for **Regional Energy Consumption Forecasting**. Designed with **Continuous / Retraining Learning methods (Batch & Online Learning)** to prevent data decay and maintain peak predictive accuracy over time.

---

## 🌟 Key Features

1. **Multi-Model Forecasting Engine**:
   - **XGBoost Regressor**: Gradient-boosted decision trees capturing non-linear relationships (achieving ~2.8% MAPE).
   - **Random Forest Regressor**: Bagged ensemble providing robust variance reduction.
   - **Online Incremental Learner (`SGDRegressor`)**: Supports real-time streaming updates via `partial_fit()` without full retraining.
   - **Adaptive Ensemble**: Dynamically weighted ensemble combining top model predictions based on out-of-sample validation performance.
   - **Holt-Winters Exponential Smoothing**: Classical statistical benchmark capturing level, trend, and seasonal components.

2. **Continuous Learning & Retraining (Zero Data Decay)**:
   - **Kolmogorov-Smirnov (KS) Statistical Drift Detection**: Detects covariate shifts and changes in energy consumption distribution.
   - **Concept Drift Monitoring**: Tracks rolling prediction error degradation (MAPE ratio).
   - **Online Streaming Updates**: 1-click or automated streaming batch ingestion updating model weights in milliseconds.
   - **Batch Retraining Trigger**: Full walk-forward re-calibration when structural drift occurs.
   - **Model Registry & Audit Log**: Records model versions, timestamps, training sizes, drift statuses, and accuracy scorecards.

3. **Advanced Time-Series Feature Engineering**:
   - **Cyclical Encodings**: $\sin/\cos$ transformations of hour-of-day, day-of-week, month, and day-of-year.
   - **Autoregressive Lags**: $t-1, t-2, t-3, t-24, t-48, t-168$ (1 hour to 1 week back).
   - **Rolling Statistics**: Moving average, standard deviation, min, and max across 6h, 24h, and 168h windows (zero-leakage shifting).
   - **Exogenous Temperature & Weather**: Heating and cooling degree demand response.

4. **Modern Interactive Dashboard (Streamlit + Plotly)**:
   - Dynamic forecast horizons (24h, 48h, 7 days, 14 days).
   - Zoomable time-series charts with $\pm 95\%$ confidence prediction intervals.
   - Diurnal load curve analysis (24-hour demand profile) & Day-of-week comparisons.
   - Hour $\times$ Day energy intensity heatmaps.
   - Feature importance rankings.

5. **Dual Export Center**:
   - 📥 **CSV Export**: Clean tabular forecast data with timestamps, predictions, and confidence bounds.
   - 📄 **Official PDF Report**: Publication-ready multi-page executive summary with KPI cards, model scorecards, embedded high-res charts, diurnal curves, and 24-hour breakdown table.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Benchmark Dataset (Optional, generated automatically)
```bash
python generate_sample_data.py
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py                      # Main Interactive Streamlit Dashboard
├── generate_sample_data.py     # Benchmark multi-region energy dataset generator
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── data/
│   └── sample_energy_data.csv  # Benchmark hourly dataset (85k+ records across 3 regions)
├── models/
│   └── registry/               # Model checkpoints and version history log
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Flexible parser, column auto-detection, resampling & outlier handling
│   ├── feature_engineering.py  # Cyclical time encodings, autoregressive lags, rolling statistics
│   ├── models.py               # XGBoost, Random Forest, Online SGD, Holt-Winters, Adaptive Ensemble
│   ├── drift_detector.py       # KS-test & rolling error concept drift monitoring
│   ├── trainer.py              # TimeSeriesSplit validation, recursive forecasting & continuous retraining
│   └── pdf_generator.py        # ReportLab multi-page executive PDF report builder with embedded charts
├── tests/
│   └── test_pipeline.py        # Automated test suite
└── exports/                    # Output directory for exported CSVs and PDFs
```

---

## 🧪 Running Automated Tests
```bash
python -m unittest discover -s tests
```
