# SMART STREET LIGHTING & ENERGY USAGE ANALYTICS SYSTEM
> **AI-powered energy monitoring, fault prediction and maintenance decision support**

A comprehensive, fully functional hackathon prototype built using **Streamlit**, **Pandas**, **NumPy**, **Plotly**, and **Scikit-learn** conforming to the finalized Software Requirements Specification (SRS).

---

## 🏛️ System Architecture & Workflow

The system faithfully executes the analytical pipeline outlined in the SRS:

```
Raw Data
   └──> Data Preprocessing
          └──> Energy Usage Analysis
                 └──> Zone Analysis
                        └──> ML Fault Prediction (Random Forest)
                               └──> Fault Alerts & Watchlist
                                      └──> Maintenance Decision Support
                                             └──> Energy Saving Analysis
                                                    └──> Interactive Dashboard
```

---

## 📂 Project Structure

```
smart_street_lighting/
├── app.py                      # Main Streamlit application with multi-page navigation & UI
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation & demo guide
├── data/
│   └── sample_street_lights.csv # Realistic demonstration dataset (650+ records)
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py        # Cleaning, missing-value handling, deduplication, dataset generator
│   ├── analytics.py            # KPI metrics, aggregations, and Plotly visualizations
│   ├── fault_prediction.py     # Random Forest model, confusion matrix, and individual risk scorer
│   ├── energy_saving.py        # Wastage detection and potential energy savings calculations
│   └── recommendations.py      # Data-driven maintenance and energy optimization advisory
└── assets/
    └── style.css               # Smart-city dark navy dashboard styling
```

---

## ⚡ Quick Start & Running the Prototype

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Install Dependencies
Open a terminal in the `smart_street_lighting` directory and run:

```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit app:

```bash
streamlit run app.py
```

The application will launch locally at `http://localhost:8501`.

---

## 🎯 19-Step Hackathon Presentation Demo Flow

Follow this exact sequential walkthrough for a winning hackathon demonstration:

1. **Open Dashboard (`🏠 Dashboard`)**:
   - Point out the modern dark navy smart-city command center aesthetic.
   - Note the **"Demo / Sample Dataset"** badge indicating historical telemetry is loaded.
2. **Show Key Performance Indicators (KPIs)**:
   - Walk through the 7 dynamic metric cards: *Total Lamps (650+)*, *Active Lamps*, *Faulty Lamps*, *Total Energy (kWh)*, *Average Energy*, *High-Risk Lamps*, and *Estimated Potential Energy Savings*.
3. **Explore Main Visualizations**:
   - Highlight the **Energy Consumption Trend**, **Zone Comparisons**, **Fault Distribution**, and **Lamp Status Distribution**.
4. **Inspect High-Risk Luminaires**:
   - Scroll to the urgent attention table identifying fixtures with high failure probabilities.
5. **Navigate to Data Upload (`📁 Data Upload & Preprocessing`)**:
   - Demonstrate the upload container supporting CSV and Excel files.
6. **Examine Preprocessing Statistics**:
   - Show how the pipeline automatically audited the dataset: Original rows, missing values detected & imputed with median/mode, duplicate rows stripped, and clean rows prepared.
7. **Download Cleaned Dataset**:
   - Click **"Download Cleaned Dataset (CSV)"** to show the ready-to-use processed export.
8. **Navigate to Energy Analytics (`⚡ Energy Analytics`)**:
   - Demonstrate dynamic filtering by Date Range, Zone, Lamp Type, and Status.
   - Show how all 4 Plotly charts update instantaneously based on active filters.
9. **Navigate to Zone Analysis (`🗺️ Zone Analysis`)**:
   - Highlight the statistical zone comparison matrix.
10. **Identify High-Energy Zones**:
    - Show the data-driven alert flagging zones exceeding the 70th percentile of energy draw (e.g., *Industrial Corridor* & *Downtown Commercial*).
11. **Drill Down into a Specific Zone**:
    - Select a zone from the dropdown to reveal localized fixture counts, fault rates, and the top 10 highest-drawing lamps.
12. **Navigate to Machine Learning Fault Prediction (`🤖 Fault Prediction`)**:
    - Review the Scikit-learn **Random Forest Classifier** performance metrics: Accuracy, Precision, Recall, and F1-Score.
    - Inspect the interactive **Confusion Matrix** and **Feature Importance** chart highlighting key failure drivers (Voltage anomalies, burn hours, fixture age).
13. **Individual Lamp Prediction**:
    - Select a specific `Lamp_ID` from the dropdown.
    - Review its live electrical telemetry (Voltage, Current, Previous Faults, Operating Hours).
14. **Predict Fault Risk**:
    - Click **`[🚀 PREDICT FAULT RISK]`**.
    - Observe the animated calculation returning **Predicted Fault Risk** percentage and the color-coded **Risk Level** (`LOW`, `MEDIUM`, `HIGH`).
    - Note the decision-support disclaimer clarifying this is an algorithmic estimate.
15. **Navigate to Fault Alerts (`🚨 Fault Alerts`)**:
    - Show the filterable high-risk luminaire watchlist.
    - Filter by Risk Level (`HIGH`) and Zone to demonstrate dispatch prioritization.
16. **Navigate to Maintenance Decision Support (`🔧 Maintenance`)**:
    - Review the multi-factor composite priority ranking (`P1 - Immediate Dispatch`, `P2 - High Priority`, `P3 - Scheduled Review`).
    - Point out specific recommended actions: *Immediate electrical inspection*, *Preventive maintenance*, *Overhaul*.
17. **Navigate to Energy Saving (`💡 Energy Saving`)**:
    - Review the 4 metrics: *Current Consumption*, *Estimated Excess Consumption*, *Estimated Potential Energy Saving*, and *Saving Percentage*.
    - Note the clear labeling: **"Estimated Potential Energy Saving"**.
18. **Explain the Energy Waterfall Chart**:
    - Walk through the interactive Plotly waterfall showing how LED retrofits and smart off-peak dimming achieve municipal energy reduction.
19. **Navigate to Recommendations (`📋 Recommendations`)**:
    - Conclude the demo by summarizing the data-driven actions separated into **Category A (Maintenance Recommendations)** and **Category B (Energy Optimization Recommendations)**.

---

## 📊 Dataset Schema & Telemetry Dictionary

| Column | Type | Description |
| :--- | :--- | :--- |
| `Lamp_ID` | String | Unique luminaire asset identifier (e.g., `SL-1001`) |
| `Zone` | Categorical | Smart city sector (e.g., Downtown, Industrial, Waterfront) |
| `Lamp_Type` | Categorical | Technology (LED Smart, HPS, Metal Halide, Solar-Hybrid) |
| `Lamp_Status` | Categorical | Operational condition (`Active`, `Inactive`, `Maintenance Required`) |
| `Operating_Hours` | Float | Cumulative lamp burning hours |
| `Energy_Consumption` | Float | Monthly power consumption in kilowatt-hours (kWh) |
| `Date_Time` | Datetime | Telemetry measurement timestamp |
| `Voltage` | Float | RMS operating voltage (Nominal 220V - 230V) |
| `Current` | Float | Operating current in amperes (A) |
| `Previous_Faults` | Integer | Historical maintenance and repair events count |
| `Lamp_Age` | Integer | Fixture operational age in months |
| `Fault_Status` | Binary | Supervised label (`0` = Normal, `1` = Fault/At Risk) |

---

## 🛡️ Key System Safeguards

- **Graceful Fallback**: If an uploaded dataset lacks the `Fault_Status` label, the system displays `"Fault prediction requires labeled Fault_Status data."` while maintaining full functionality across all other 8 analytics pages.
- **Data-Driven Benchmarks**: High-consumption zones and excessive operating hours are determined dynamically using distribution percentiles (e.g., 70th/85th percentiles) rather than rigid arbitrary constants.
- **Decision-Support Wording**: The system clearly distinguishes advisory statistical models from real-world guarantees.
