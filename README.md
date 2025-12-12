# Forecasting with AutoGluon on music session data

## 1. General Description

This project builds an end‑to‑end forecasting pipeline for user activity derived from the LastFM dataset. The workflow identifies the *top 1 user* with the highest number of total sessions and constructs a daily time‑series for that user. After preprocessing and ensuring full date continuity, the project forecasts the next **3 months (90 days)** of the selected metric — *number_of_sessions* — starting from the last available historical record.

The pipeline includes:
- Extracting, transforming, and cleaning raw TSV data using PySpark
- Identifying the most active user based on session counts
- Filling missing dates to create a continuous daily series
- Training AutoGluon forecasting models (DeepAR and AutoARIMA)
- Generating quantile‑based predictions for the next 90 days
- Saving raw and post‑processed forecasts as CSV files

---

## 2. Environment Setup

Follow these steps to create and activate your environment:


#### Step 1: Navigate to the project directory
```bash
cd /path/to/your/project
```
#### Step 2: Create a virtual environment (Python 3.11 recommended)
```bash
python3.11 -m venv myvenv
```
#### Step 3: Activate the virtual environment
```bash
source myvenv/bin/activate  # On macOS/Linux
# For Windows: myvenv\Scripts\activate
```
#### Step 4: Upgrade pip
```bash
pip install --upgrade pip
```
#### Step 5: Install dependencies
```bash
pip install -r requirements.txt
```
Make sure Python 3.11 is installed (AutoGluon requires Python version 3.9, 3.10, 3.11, or 3.12)

## 3. Executing the Forecasting Pipeline
You can execute the full pipeline (data processing + forecasting) using:

```bash
python main.py -i "/path/to/input/.tsv" -o "/path/to/output/"
```
Where:

* -i specifies the input TSV file from LastFM
* -o specifies the output directory where results will be saved

The application expects the .tsv file to be present in the input path.

The pipeline will:

a. Convert the input to Parquet

b. Generate the top user’s time series

c. Fill missing dates and validate continuity

d. Save the processed series as .csv (TSV-separated)

e. Train forecasting models (AutoARIMA + DeepAR)

Predict the next 90 days and output results as:
* forecast_output.csv (raw forecast)
* postprocessed_forecast_output.csv (rounded, renamed columns)

## 4. Sample output
- The sample output can be viewed in-
```bash
lastfm-forecasting-autogluon/
├── sample_output
```

## 5. Additional Notes
- The project assumes daily frequency time series.
- Quantile forecasts (p_30, p_50, p_90) are supported and saved.
- The best model is stored as best_model.pkl and can be reused without retraining.
- All scripts are modular and orchestrated via main.py.