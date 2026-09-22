"""
Comprehensive Integration & Verification Test Suite
Tests all modules: Preprocessing, Analytics, ML Fault Prediction, Energy Saving, Recommendations.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from modules.preprocessing import generate_sample_dataset, preprocess_data, get_dataset_summary
from modules.analytics import (
    calculate_dashboard_kpis,
    get_energy_overview_stats,
    get_zone_summary_table,
    get_single_zone_details,
    plot_energy_trend,
    plot_energy_by_zone,
    plot_fault_distribution,
    plot_lamp_status_distribution
)
from modules.fault_prediction import (
    train_fault_prediction_model,
    predict_single_lamp,
    score_dataset_fault_risks
)
from modules.energy_saving import analyze_energy_savings, plot_energy_savings_waterfall
from modules.recommendations import generate_recommendations

def test_pipeline():
    print("=" * 60)
    print("STARTING TEST SUITE: Smart Street Lighting System")
    print("=" * 60)

    # 1. Test Dataset Generation
    print("\n[1/5] Testing Sample Dataset Generator...")
    df_raw = generate_sample_dataset(n_records=650)
    assert len(df_raw) >= 650, f"Expected >= 650 records, got {len(df_raw)}"
    print(f"  ✓ Generated {len(df_raw)} realistic records with duplicates & missing values.")

    # 2. Test Preprocessing
    print("\n[2/5] Testing Preprocessing Pipeline...")
    summary = get_dataset_summary(df_raw)
    assert summary['duplicates_count'] > 0, "Expected duplicates in raw data"
    assert summary['total_missing'] > 0, "Expected missing values in raw data"
    
    df_clean, report = preprocess_data(df_raw)
    assert report['duplicates_removed'] > 0, "Duplicates were not removed"
    assert df_clean.isnull().sum().sum() == 0, "Missing values were not completely imputed"
    print(f"  ✓ Preprocessing complete: {report['original_records']} -> {report['cleaned_records']} clean rows.")
    print(f"  ✓ Duplicates removed: {report['duplicates_removed']}, Missing values imputed: {report['missing_values_handled']}.")

    # 3. Test Analytics
    print("\n[3/5] Testing Analytics & KPI Calculations...")
    kpis = calculate_dashboard_kpis(df_clean)
    assert kpis['total_lamps'] == len(df_clean), "KPI Total lamps mismatch"
    assert kpis['total_energy'] > 0, "KPI Total energy must be > 0"
    print(f"  ✓ KPIs computed: Total Lamps={kpis['total_lamps']}, Total Energy={kpis['total_energy']} kWh, Savings={kpis['potential_savings']} kWh.")

    zone_table, high_zones = get_zone_summary_table(df_clean)
    assert not zone_table.empty, "Zone summary table should not be empty"
    print(f"  ✓ Zone analysis: Found {len(zone_table)} zones. High consumption zones: {high_zones}")

    fig_trend = plot_energy_trend(df_clean)
    fig_zone = plot_energy_by_zone(df_clean)
    fig_fault = plot_fault_distribution(df_clean)
    fig_status = plot_lamp_status_distribution(df_clean)
    assert fig_trend is not None and fig_zone is not None, "Plotly chart generation failed"
    print("  ✓ Plotly charts successfully generated.")

    # 4. Test ML Fault Prediction
    print("\n[4/5] Testing ML Fault Prediction Pipeline...")
    pipeline, metrics, cm_fig, fi_fig, err = train_fault_prediction_model(df_clean)
    assert err is None, f"ML training failed with error: {err}"
    assert pipeline is not None, "Pipeline was not created"
    assert 'accuracy' in metrics and metrics['accuracy'] > 60.0, f"Low accuracy: {metrics}"
    print(f"  ✓ Random Forest trained: Accuracy={metrics['accuracy']}%, Precision={metrics['precision']}%, Recall={metrics['recall']}%, F1={metrics['f1']}%.")

    # Test single lamp prediction
    sample_lamp = df_clean.iloc[0].to_dict()
    prob, risk, color = predict_single_lamp(pipeline, sample_lamp)
    assert 0.0 <= prob <= 100.0, f"Invalid probability: {prob}"
    assert risk in ['LOW', 'MEDIUM', 'HIGH'], f"Invalid risk level: {risk}"
    print(f"  ✓ Single lamp prediction for {sample_lamp['Lamp_ID']}: Prob={prob}%, Risk={risk}.")

    # Test scoring full dataset
    df_scored = score_dataset_fault_risks(pipeline, df_clean)
    assert 'Fault_Probability' in df_scored.columns, "Fault_Probability column missing"
    assert 'Risk_Level' in df_scored.columns, "Risk_Level column missing"
    print(f"  ✓ Scored all {len(df_scored)} lamps for fault risks.")

    # Test graceful fallback when Fault_Status is missing
    df_no_target = df_clean.drop(columns=['Fault_Status'])
    _, _, _, _, expected_err = train_fault_prediction_model(df_no_target)
    assert expected_err == "Fault prediction requires labeled Fault_Status data.", f"Unexpected fallback error: {expected_err}"
    print(f"  ✓ Graceful fallback verified: '{expected_err}'")

    # 5. Test Energy Savings & Recommendations
    print("\n[5/5] Testing Energy Savings & Recommendations...")
    savings = analyze_energy_savings(df_clean)
    assert savings['current_consumption'] > 0, "Current consumption should be > 0"
    assert savings['potential_saving'] > 0, "Potential savings should be > 0"
    print(f"  ✓ Energy savings calculated: Current={savings['current_consumption']} kWh, Est. Potential Saving={savings['potential_saving']} kWh ({savings['saving_percentage']}%).")

    recs = generate_recommendations(df_clean, df_scored, savings)
    assert len(recs['maintenance_recs']) > 0, "Expected maintenance recommendations"
    assert len(recs['energy_recs']) > 0, "Expected energy optimization recommendations"
    print(f"  ✓ Generated {len(recs['maintenance_recs'])} Maintenance Recommendations and {len(recs['energy_recs'])} Energy Optimization Recommendations.")

    print("\n" + "=" * 60)
    print("ALL 5 MODULE TEST SUITES PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
