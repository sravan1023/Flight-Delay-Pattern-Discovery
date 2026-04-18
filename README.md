# CMPE-255 Flight Delay Pattern Discovery

Analyze U.S. flight delay data to uncover patterns across airlines, airports, seasons, and delay causes using EDA, clustering, and predictive modeling.

## Dataset

**Source**: [Bureau of Transportation Statistics — Airline On-Time Performance](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp)  
**File**: `dataset/Airline_Delay_Cause.csv` (~135K records, 21 columns)

Each row represents one airline at one airport for one month, with columns for:
- Flight counts (arrivals, delayed >15 min, cancelled, diverted)
- Delay causes (carrier, weather, NAS, security, late aircraft) — both count and minutes
- Time identifiers (year, month)

## Project Structure

```
├── data_preprocessing.ipynb     # Cleaning, feature engineering, aggregation
├── eda.ipynb                    # Exploratory data analysis & visualizations
├── clustering_analysis.ipynb    # KMeans clustering of airports & airlines
├── predictive_modeling.ipynb    # Baseline & secondary model comparison
├── dataset/
│   ├── Airline_Delay_Cause.csv  # Raw data
│   ├── cleaned_flight_data.csv  # Cleaned + engineered features
│   ├── airport_metrics.csv      # Airport-level aggregation
│   ├── airline_metrics.csv      # Airline-level aggregation
│   └── monthly_trends.csv       # Monthly trend aggregation
├── outputs/
│   ├── figures/                 # All plots (PNG)
│   ├── tables/                  # Cluster summaries, model results (CSV)
│   └── models/                  # Saved KMeans & best ML models (PKL)
└── requirements.txt
```

## Notebooks

### 1. `data_preprocessing.ipynb`
- Loads raw CSV, inspects missing values and duplicates
- Renames columns, converts types, removes invalid rows
- Engineers features: delay_rate, cancellation_rate, season, quarter, dominant_delay_cause, delay cause percentages
- Aggregates to airport-level, airline-level, monthly, and seasonal views
- Saves cleaned data to `dataset/`

### 2. `eda.ipynb`
- Overall delay landscape (% delayed, cancelled, diverted)
- Yearly and monthly delay trends
- Seasonal impact with delay cause composition shifts
- Root cause breakdown (bar + pie)
- Airline performance comparison (delay rate + severity)
- Airport hotspot analysis (volume vs delay rate scatter)
- Delay cause profiles by airline
- Correlation heatmap
- Year-over-year delay cause shift
- High-delay route identification

### 3. `clustering_analysis.ipynb`
- Builds feature matrices for airports and airlines
- Selects optimal K via silhouette score + elbow method
- Runs KMeans, auto-labels cluster archetypes (e.g., "High-Delay / High-Severity / Low-Volume")
- Visualizes clusters with scatter plots and profile comparisons
- Identifies worst/best cluster members
- Analyzes delay cause fingerprint per cluster

### 4. `predictive_modeling.ipynb`
- **Baseline**: Logistic Regression, Decision Tree (classification) / Ridge, Decision Tree (regression)
- **Secondary**: Random Forest, Gradient Boosting (both tasks)
- Full evaluation: Accuracy, Precision, Recall, F1, MAE, RMSE, R²
- Confusion matrices, actual vs predicted plots
- Head-to-head comparison charts
- Feature importance from best models
- 5-fold cross-validation

## Setup

```bash
pip install -r requirements.txt
```

Then open and run each notebook in order (1 -> 4). Each notebook reads outputs from the previous one.

## Key Results

- **Late Aircraft** and **Carrier** delays are the dominant causes (~70%+ combined)
- Summer and winter months show the highest delay rates
- Ensemble models (Random Forest, Gradient Boosting) significantly outperform baselines
- Airport clustering reveals distinct archetypes: high-volume hubs with moderate delays vs small airports with extreme delay rates
