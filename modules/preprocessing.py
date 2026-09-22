"""
Preprocessing Module
Handles dataset loading, validation, missing value imputation, duplicate removal,
data type normalization, and realistic sample dataset generation.
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta

EXPECTED_COLUMNS = [
    'Lamp_ID',
    'Zone',
    'Lamp_Type',
    'Lamp_Status',
    'Operating_Hours',
    'Energy_Consumption',
    'Date_Time',
    'Voltage',
    'Current',
    'Previous_Faults',
    'Lamp_Age',
    'Fault_Status'
]

NUMERIC_COLUMNS = [
    'Operating_Hours',
    'Energy_Consumption',
    'Voltage',
    'Current',
    'Previous_Faults',
    'Lamp_Age'
]

CATEGORICAL_COLUMNS = [
    'Zone',
    'Lamp_Type',
    'Lamp_Status'
]


def load_data(file_or_path):
    """
    Loads data from a file path or an uploaded file object (Streamlit UploadedFile).
    Supports CSV and Excel formats.
    """
    if hasattr(file_or_path, 'name'):
        # Uploaded file object
        filename = file_or_path.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file_or_path)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_or_path)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel (.xlsx/.xls) file.")
    elif isinstance(file_or_path, str):
        # File path string
        filepath = file_or_path.lower()
        if filepath.endswith('.csv'):
            df = pd.read_csv(file_or_path)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_or_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    elif isinstance(file_or_path, pd.DataFrame):
        df = file_or_path.copy()
    else:
        raise ValueError("Invalid input provided for data loading.")

    return df


def get_dataset_summary(df):
    """
    Computes summary metrics for raw dataset before and after cleaning:
    - Original Records
    - Missing Values Count per column and total
    - Duplicate Records Count
    - Memory usage
    """
    total_records = len(df)
    duplicates_count = int(df.duplicated().sum())
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    return {
        'total_records': total_records,
        'duplicates_count': duplicates_count,
        'missing_by_col': missing_by_col,
        'total_missing': total_missing,
        'columns': list(df.columns)
    }


def preprocess_data(df):
    """
    Preprocesses raw dataset:
    - Normalizes column names (strip, capitalize correctly)
    - Detects and removes duplicates
    - Imputes missing values:
        - Numerical columns with median
        - Categorical columns with mode
        - Lamp_ID with synthetic IDs if missing
        - Date_Time with forward-fill or generated dates
    - Converts data types (datetime, numeric, categorical)
    - Validates presence of essential columns
    
    Returns: (cleaned_df, cleaning_report_dict)
    """
    cleaned_df = df.copy()
    original_count = len(cleaned_df)
    
    # 1. Normalize column names (strip spaces, match case-insensitively to standard names)
    col_mapping = {}
    standard_cols_lower = {col.lower(): col for col in EXPECTED_COLUMNS}
    for col in cleaned_df.columns:
        norm = col.strip().replace(' ', '_').lower()
        if norm in standard_cols_lower:
            col_mapping[col] = standard_cols_lower[norm]
        else:
            col_mapping[col] = col.strip()
    cleaned_df.rename(columns=col_mapping, inplace=True)
    
    # 2. Track initial missing & duplicates
    initial_missing = int(cleaned_df.isnull().sum().sum())
    duplicate_count = int(cleaned_df.duplicated().sum())
    
    # 3. Remove duplicate records
    cleaned_df.drop_duplicates(inplace=True)
    records_after_dedup = len(cleaned_df)
    
    # 4. Handle Lamp_ID
    if 'Lamp_ID' not in cleaned_df.columns:
        cleaned_df['Lamp_ID'] = [f"SL-{1000 + i}" for i in range(len(cleaned_df))]
    else:
        cleaned_df['Lamp_ID'] = cleaned_df['Lamp_ID'].fillna(
            pd.Series([f"SL-UNK-{i}" for i in range(len(cleaned_df))])
        ).astype(str)
    
    # 5. Handle Date_Time
    if 'Date_Time' in cleaned_df.columns:
        cleaned_df['Date_Time'] = pd.to_datetime(cleaned_df['Date_Time'], errors='coerce')
        if cleaned_df['Date_Time'].isnull().any():
            cleaned_df['Date_Time'] = cleaned_df['Date_Time'].ffill().bfill()
            if cleaned_df['Date_Time'].isnull().any():
                cleaned_df['Date_Time'] = pd.date_range(end=datetime.now(), periods=len(cleaned_df), freq='H')
    else:
        cleaned_df['Date_Time'] = pd.date_range(end=datetime.now(), periods=len(cleaned_df), freq='H')
        
    cleaned_df['Date'] = cleaned_df['Date_Time'].dt.date
    cleaned_df['Hour'] = cleaned_df['Date_Time'].dt.hour
    
    # 6. Impute Categorical Columns
    for col in CATEGORICAL_COLUMNS:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].empty else 'Unknown'
            cleaned_df[col] = cleaned_df[col].replace({'nan': mode_val, 'None': mode_val, '': mode_val})
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
        else:
            if col == 'Zone':
                cleaned_df['Zone'] = 'General Zone'
            elif col == 'Lamp_Type':
                cleaned_df['Lamp_Type'] = 'LED (Smart)'
            elif col == 'Lamp_Status':
                cleaned_df['Lamp_Status'] = 'Active'

    # 7. Impute & Convert Numeric Columns
    default_medians = {
        'Operating_Hours': 4500.0,
        'Energy_Consumption': 45.0,
        'Voltage': 225.0,
        'Current': 0.8,
        'Previous_Faults': 0,
        'Lamp_Age': 24
    }
    
    for col in NUMERIC_COLUMNS:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            median_val = cleaned_df[col].median()
            if pd.isna(median_val):
                median_val = default_medians.get(col, 0.0)
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
        else:
            cleaned_df[col] = default_medians.get(col, 0.0)
            
    # Round integer columns properly
    cleaned_df['Previous_Faults'] = cleaned_df['Previous_Faults'].round().astype(int)
    cleaned_df['Lamp_Age'] = cleaned_df['Lamp_Age'].round().astype(int)
    cleaned_df['Operating_Hours'] = cleaned_df['Operating_Hours'].round(1)
    cleaned_df['Energy_Consumption'] = cleaned_df['Energy_Consumption'].round(2)
    cleaned_df['Voltage'] = cleaned_df['Voltage'].round(1)
    cleaned_df['Current'] = cleaned_df['Current'].round(2)

    # 8. Check Fault_Status if present
    has_fault_status = 'Fault_Status' in cleaned_df.columns
    if has_fault_status:
        cleaned_df['Fault_Status'] = pd.to_numeric(cleaned_df['Fault_Status'], errors='coerce')
        cleaned_df['Fault_Status'] = cleaned_df['Fault_Status'].fillna(0).round().astype(int)
        # Ensure binary 0 or 1
        cleaned_df['Fault_Status'] = cleaned_df['Fault_Status'].apply(lambda x: 1 if x >= 1 else 0)

    # Reset index
    cleaned_df.reset_index(drop=True, inplace=True)
    cleaned_count = len(cleaned_df)
    
    cleaning_report = {
        'original_records': original_count,
        'duplicates_removed': duplicate_count,
        'missing_values_handled': initial_missing,
        'cleaned_records': cleaned_count,
        'has_fault_status': has_fault_status
    }
    
    return cleaned_df, cleaning_report


def generate_sample_dataset(n_records=650, seed=42):
    """
    Generates a realistic Smart Street Lighting dataset with at least 650 records.
    Contains intentional realistic edge cases (occasional missing values, duplicates,
    and telemetry anomalies) to demonstrate preprocessing and ML.
    """
    np.random.seed(seed)
    
    zones = [
        'Downtown Commercial',
        'North Residential',
        'Industrial Corridor',
        'Tech Park Sector 5',
        'Harbor Waterfront',
        'Suburban Ring'
    ]
    zone_weights = [0.22, 0.20, 0.18, 0.15, 0.12, 0.13]
    
    lamp_types = [
        'LED (Smart)',
        'HPS (High Pressure Sodium)',
        'Metal Halide',
        'Solar-Hybrid LED'
    ]
    lamp_type_weights = [0.50, 0.25, 0.15, 0.10]
    
    # Base parameters per lamp type: (mean_power_kwh_per_month, current_amp_mean)
    type_profiles = {
        'LED (Smart)': {'kwh': 28.0, 'amp': 0.38, 'failure_bias': 0.08},
        'HPS (High Pressure Sodium)': {'kwh': 82.0, 'amp': 1.15, 'failure_bias': 0.24},
        'Metal Halide)': {'kwh': 90.0, 'amp': 1.25, 'failure_bias': 0.28},
        'Solar-Hybrid LED': {'kwh': 14.0, 'amp': 0.22, 'failure_bias': 0.05}
    }
    
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    for i in range(n_records):
        lamp_id = f"SL-{1001 + i}"
        zone = np.random.choice(zones, p=zone_weights)
        lamp_type = np.random.choice(lamp_types, p=lamp_type_weights)
        
        # Age in months
        if lamp_type in ['HPS (High Pressure Sodium)', 'Metal Halide']:
            age = int(np.random.randint(24, 72))
            op_hours = float(np.round(np.random.normal(9500, 2500), 1))
        else:
            age = int(np.random.randint(4, 40))
            op_hours = float(np.round(np.random.normal(4800, 1800), 1))
        op_hours = max(250.0, op_hours)
        
        # Previous faults
        if age > 40 or op_hours > 10000:
            prev_faults = int(np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.35, 0.2, 0.1, 0.05]))
        else:
            prev_faults = int(np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05]))
            
        # Voltage: nominal ~225V, with realistic anomalies
        is_voltage_anomaly = np.random.rand() < 0.09
        if is_voltage_anomaly:
            voltage = float(np.round(np.random.choice([
                np.random.uniform(175.0, 194.0), # Sag
                np.random.uniform(246.0, 260.0)  # Surge
            ]), 1))
        else:
            voltage = float(np.round(np.normal(225.0, 4.5), 1))
            
        # Profile calculation
        profile = type_profiles.get(lamp_type, {'kwh': 35.0, 'amp': 0.5, 'failure_bias': 0.1})
        # Energy consumption correlated with operating hours, lamp type, zone
        zone_mult = 1.25 if zone in ['Downtown Commercial', 'Industrial Corridor'] else (0.85 if zone == 'Tech Park Sector 5' else 1.0)
        base_kwh = profile['kwh'] * zone_mult
        energy = float(np.round(max(5.0, np.random.normal(base_kwh, base_kwh * 0.18)), 2))
        
        # Current correlated with voltage and power
        current = float(np.round(max(0.1, np.random.normal(profile['amp'], 0.06)), 2))
        if voltage < 195.0: # Compensatory current surge in ballasts
            current = float(np.round(current * 1.35, 2))
            
        # Timestamp
        random_days = np.random.randint(0, 90)
        random_hours = np.random.randint(0, 24)
        timestamp = (start_date + timedelta(days=random_days, hours=random_hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Fault probability calculation for realistic ground-truth label
        risk_score = (
            profile['failure_bias'] * 0.35 +
            (1.0 if (voltage < 195 or voltage > 245) else 0.0) * 0.30 +
            (prev_faults / 4.0) * 0.20 +
            min(1.0, op_hours / 15000.0) * 0.15 +
            (0.15 if zone == 'Harbor Waterfront' else 0.0) # Marine environment
        )
        fault_prob = min(0.95, max(0.02, risk_score + np.random.normal(0, 0.05)))
        fault_status = 1 if np.random.rand() < fault_prob else 0
        
        # Status
        if fault_status == 1:
            status = np.random.choice(['Active', 'Maintenance Required', 'Inactive'], p=[0.35, 0.55, 0.10])
        else:
            status = np.random.choice(['Active', 'Inactive'], p=[0.94, 0.06])
            
        data.append({
            'Lamp_ID': lamp_id,
            'Zone': zone,
            'Lamp_Type': lamp_type,
            'Lamp_Status': status,
            'Operating_Hours': op_hours,
            'Energy_Consumption': energy,
            'Date_Time': timestamp,
            'Voltage': voltage,
            'Current': current,
            'Previous_Faults': prev_faults,
            'Lamp_Age': age,
            'Fault_Status': fault_status
        })
        
    df = pd.DataFrame(data)
    
    # Inject ~12 duplicate rows and ~18 missing values to demonstrate data cleaning capabilities
    dup_indices = np.random.choice(range(len(df)), size=12, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Missing values in random numeric & categorical spots
    for col in ['Voltage', 'Operating_Hours', 'Current', 'Lamp_Status']:
        nan_indices = np.random.choice(range(len(df)), size=5, replace=False)
        df.loc[nan_indices, col] = np.nan
        
    return df
