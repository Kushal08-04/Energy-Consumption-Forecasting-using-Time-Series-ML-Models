# ⚡ GridCast AI: Energy Consumption Forecasting

An end-to-end machine-learning platform for **short-term regional electricity-demand forecasting**. GridCast AI combines data ingestion, preprocessing, advanced time-series feature engineering, multi-model forecasting, continuous learning, drift monitoring, model versioning, interactive visualization, and CSV/PDF reporting in a single Python application.

> **B.Tech 3rd Year Project — Department of CSE, Data Science, ABES Engineering College, Ghaziabad**  
> **Author:** Kushal Chaudhary

---

## 🌟 Key Features

### 1. Multi-Model Forecasting Engine

- **XGBoost Regressor** — gradient-boosted trees for nonlinear load patterns; the benchmark implementation reports approximately **2.8% MAPE**.
- **Random Forest Regressor** — ensemble model providing robust variance reduction.
- **Online Incremental Learner (`SGDRegressor`)** — supports incremental updates through `partial_fit()` without full model retraining.
- **Holt-Winters Exponential Smoothing (ETS)** — statistical benchmark for level, trend, and daily seasonality.
- **Adaptive Ensemble** — combines XGBoost, Random Forest, and SGD predictions using weights derived from validation performance.

### 2. Continuous Learning & Retraining

- **KS statistical drift detection** for changes in the feature/data distribution.
- **Concept-drift monitoring** using rolling forecast-error degradation.
- **Online updates** for models that support incremental learning.
- **Batch retraining** for complete recalibration when structural drift is detected.
- **Model registry and audit log** recording model versions, timestamps, training sizes, drift states, and scorecards.

### 3. Advanced Time-Series Feature Engineering

- Cyclical `sin/cos` encodings for hour, weekday, month, and day-of-year.
- Autoregressive lags: `t-1`, `t-2`, `t-3`, `t-24`, `t-48`, `t-168`.
- Rolling mean, standard deviation, minimum, and maximum over 6-hour, 24-hour, and 168-hour windows.
- Weekend, peak-hour, quarter, and other calendar indicators.
- Optional temperature-derived heating and cooling degree features.
- Leakage-safe rolling features shifted so the current target is never included in its own feature window.

### 4. Flexible Data Ingestion

The application can use its built-in benchmark dataset or accept a custom **CSV / Excel** file. The loader automatically detects common names for datetime, energy/load/demand target, region/location, and temperature, then performs sorting, duplicate removal, hourly resampling, time-based interpolation, and IQR-based outlier clipping.

### 5. Interactive Forecasting Dashboard

The Streamlit + Plotly dashboard provides configurable forecast horizons (**24, 48, 168, and 336 hours**), 90% or 95% prediction intervals, actual-vs-forecast visualization, projected peak demand, diurnal profiles, weekday comparisons, hour × weekday heatmaps, XGBoost feature importance, drift diagnostics, model scorecards, and version history.

### 6. Export Center

- **CSV export** containing timestamps, predictions, and uncertainty bounds.
- **Multi-page PDF report** containing KPI cards, model scorecards, charts, demand patterns, and a 24-hour forecast breakdown.

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│ Presentation Layer                                          │
│ Streamlit + Plotly (`app.py`)                               │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ Forecasting Logic                                            │
│ data_loader → feature_engineering → models → trainer         │
│                                   ↘ drift_detector            │
│                                   ↘ pdf_generator             │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ File-Based Storage                                           │
│ data/*.csv · models/registry/*.json · exports/*              │
└──────────────────────────────────────────────────────────────┘
```

The current application runs as a single Python process and does not require a database or external API.

---

## 📁 Project Structure

```text
Energy Consumption Forecast/
│
├── app.py
├── generate_sample_data.py
├── test_export.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_energy_data.csv
│
├── models/
│   └── registry/
│       └── version_history.json
│
├── exports/
│   └── Generated CSV and PDF reports
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── trainer.py
│   ├── drift_detector.py
│   └── pdf_generator.py
│
└── tests/
    ├── test_pipeline.py
    └── test_custom_upload.py
```

### Main modules

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard and application orchestration |
| `generate_sample_data.py` | Generates the benchmark multi-region dataset |
| `data_loader.py` | Detection, cleaning, resampling, interpolation, and outlier handling |
| `feature_engineering.py` | Calendar, lag, rolling, and weather-derived features |
| `models.py` | XGBoost, Random Forest, Online SGD, Holt-Winters, and Adaptive Ensemble |
| `trainer.py` | Training, evaluation, recursive forecasting, online update, and model registry |
| `drift_detector.py` | KS-test drift and rolling error-degradation monitoring |
| `pdf_generator.py` | Multi-page ReportLab PDF report generation |
| `test_pipeline.py` | End-to-end pipeline tests |
| `test_custom_upload.py` | Custom-column upload tests |

---

## 🚀 Installation

### Requirements

- Python **3.10+** (developed on Python 3.13)
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd "Energy Consumption Forecast"
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install scipy openpyxl
```

> `scipy` is used by the KS drift test and `openpyxl` is required for Excel uploads. Adding both to `requirements.txt` is recommended for reproducible installation.

---

## ▶️ Usage

### Launch the dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

On first load, the application can generate the benchmark dataset if it is missing and train the available models.

### Dashboard workflow

1. **Control Center** — choose data source, region, forecast horizon, confidence level, and active model, then train/benchmark.
2. **Forecast & Trajectory** — inspect historical demand, forecast values, uncertainty bands, projected peak, and test-window performance.
3. **Seasonality & Demand Patterns** — analyze hourly profiles, weekdays, heatmaps, and feature importance.
4. **Continuous Learning & Drift Monitor** — inspect drift diagnostics, scorecards, distribution comparisons, version history, and trigger online or batch updates.
5. **Export Center** — download the forecast CSV or executive PDF report.

---

## 📊 Input Data Format

A custom dataset requires at least one datetime/timestamp column and one numeric energy-consumption/load/demand column. Hourly or finer-grained data is expected, with roughly **500+ records** recommended for a useful training run.

| Field | Required | Example names |
|---|---|---|
| Datetime | Yes | `timestamp`, `datetime`, `date`, `time`, `ts`, `date_time` |
| Target | Yes | `energy_consumption_mw`, `consumption`, `load`, `energy`, `mw`, `kwh`, `power`, `demand`, `value` |
| Region | No | `region`, `area`, `zone`, `location`, `city`, `grid` |
| Temperature | No | `temperature_c`, `temperature`, `temp` |

If automatic detection is insufficient, provide explicit column names through `EnergyDataLoader(target_col="...", datetime_col="...")`.

---

## ⚙️ How It Works

### 1. Data preprocessing

```text
Sort
  ↓
Remove duplicate timestamps
  ↓
Resample to hourly frequency
  ↓
Interpolate missing periods
  ↓
Clip extreme values using IQR
```

### 2. Feature engineering

The feature pipeline combines calendar features, cyclical encodings, lag features, rolling statistics, and optional temperature features.

Lags:

```text
1h, 2h, 3h, 24h, 48h, 168h
```

Rolling windows:

```text
6h, 24h, 168h
```

Rolling features are shifted by one step to prevent target leakage.

### 3. Model training

| Model | Description |
|---|---|
| XGBoost | Gradient-boosted decision trees |
| Random Forest | Bagged tree ensemble |
| Online SGD | Incremental learner using `partial_fit()` |
| Holt-Winters | Statistical level/trend/seasonality baseline |
| Adaptive Ensemble | Validation-weighted combination of XGBoost, Random Forest, and SGD |

### 4. Evaluation

The final selected test window is chronological rather than randomly shuffled. Models are scored using **MAE, RMSE, MAPE, and R²**, ranked by MAPE, and recorded in the model registry.

### 5. Forecasting

Forecasting is recursive: each prediction is appended to a buffer so later forecast steps can use earlier predictions as lag inputs. Prediction intervals are calculated from model error and forecast step using the selected 90% or 95% confidence level.

---

## 🔄 Continuous Learning & Drift Detection

### Data drift

The platform uses the **Kolmogorov-Smirnov (KS) test** to compare a reference window against recent data.

| Status | Condition |
|---|---|
| **Drift Detected** | KS p-value < 0.01 and KS statistic > 0.15, or recent MAPE > 1.35 × baseline MAPE |
| **Warning: Minor Shift** | KS statistic > 0.10 |
| **Healthy** | Otherwise |
| **Calibrated** | Displayed after retraining or an online update |

Default windows:

```text
Reference window → 720 hours
Recent window    → 168 hours
```

### Online update

Models supporting incremental learning can process new batches using `partial_fit()`. This applies to the SGD model and the SGD component of the ensemble.

### Batch retraining

Batch retraining performs a full model rebuild and re-benchmarking using the latest available data.

### Model registry

Training and update events are recorded in:

```text
models/registry/version_history.json
```

The registry records information such as model version, timestamp, training size, accuracy metrics, drift state, and update/retraining event.

---

## 🧪 Testing

Run the automated test suite:

```bash
python -m unittest discover -s tests
```

| Test | Coverage |
|---|---|
| `tests/test_pipeline.py` | Data loader, features, models, online update, drift detector, train → forecast → update → PDF workflow |
| `tests/test_custom_upload.py` | Custom datasets with non-standard column names |

Generate sample export files with:

```bash
python test_export.py
```

---

## 🐍 Python Usage

```python
import pandas as pd
from src.data_loader import EnergyDataLoader
from src.trainer import EnergyForecastingPipeline

df = pd.read_csv("data/sample_energy_data.csv")
processed, meta = EnergyDataLoader().process(
    df,
    selected_region="North-Metro"
)

pipe = EnergyForecastingPipeline()
result = pipe.train_and_evaluate(
    processed,
    test_horizon=48,
    region_name="North-Metro"
)

print(result["scorecard"])

forecast = pipe.forecast_future(
    processed,
    horizon_steps=48,
    confidence_level=0.95
)

print(forecast.head())
```

---

## 📤 Command-Line Helpers

Regenerate the benchmark dataset:

```bash
python generate_sample_data.py
```

Train, forecast, and generate sample CSV/PDF outputs:

```bash
python test_export.py
```

---

## ⚙️ Configuration Reference

| Setting | Location | Default |
|---|---|---|
| Forecast horizon | Dashboard sidebar | 48 h |
| Horizon options | Dashboard sidebar | 24, 48, 168, 336 h |
| Confidence level | Dashboard sidebar | 95% |
| Alternative confidence | Dashboard sidebar | 90% |
| Active model | Dashboard sidebar | Adaptive Ensemble |
| Lags | `TimeSeriesFeatureEngineer` | `[1,2,3,24,48,168]` |
| Rolling windows | `TimeSeriesFeatureEngineer` | `[6,24,168]` |
| Reference drift window | `EnergyDriftDetector` | 720 h |
| Recent drift window | `EnergyDriftDetector` | 168 h |
| Drift p-value threshold | `EnergyDriftDetector` | 0.01 |
| Drift KS threshold | `EnergyDriftDetector` | 0.15 |
| MAPE degradation ratio | `EnergyDriftDetector` | 1.35 |
| Outlier clipping | `EnergyDataLoader.process` | 3.0 × IQR |
| Registry | `EnergyForecastingPipeline` | `models/registry/` |

---

## 📈 Benchmark Dataset

The built-in dataset is a synthetic multi-region hourly benchmark for demonstrating forecasting, model comparison, drift monitoring, and custom-upload workflows.

---

## ⚠️ Known Limitations

- Holt-Winters can behave poorly for long recursive forecasts; other models are preferable for multi-hour horizons.
- Some chart/report labels may display "95%" even when 90% is selected; the underlying bounds use the selected confidence level.
- A fallback KPI value is present when an active model is missing from the scorecard.
- The current **Ingest Stream Batch** workflow is demonstrative and re-feeds the last 24 hours of existing data rather than connecting to a live streaming source.
- After retraining, the UI may show **Calibrated** without independently re-running the underlying drift measurement.
- Future temperature is approximated from the previous day because a live weather-forecast feed is not integrated.
- Forecasts and PDFs are recomputed during Streamlit reruns; caching could improve performance.
- The current implementation is single-user and stores the model registry in a shared local JSON file.
- `app.py` contains a few unused imports that can be removed during cleanup.

---

## 🚀 Future Work

- Integrate a live weather-forecast API for future temperature features.
- Trigger automated retraining when drift thresholds are exceeded.
- Add deep-learning models such as LSTM and Temporal Fusion Transformer.
- Cache forecasts and reports for faster dashboard performance.
- Persist complete trained model artifacts in the registry.
- Add authentication and multi-user support.
- Containerize the application and deploy it to cloud infrastructure.
- Add production streaming ingestion and real-time energy data sources.

---

## 🛠️ Tech Stack

**Language & Application**
- Python
- Streamlit
- Plotly

**Data & ML**
- pandas
- NumPy
- scikit-learn
- XGBoost
- statsmodels
- SciPy

**Reporting**
- Matplotlib
- ReportLab

**Testing & Tooling**
- Python `unittest`
- File-based model registry

---

## 📄 License

Add the repository license that matches your intended project usage.

---

## 👤 Author

**Kushal Chaudhary**  
B.Tech — CSE, Data Science  
ABES Engineering College, Ghaziabad
