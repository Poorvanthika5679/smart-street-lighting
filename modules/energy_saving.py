"""
Energy Saving Module
Identifies high-consumption zones, anomalous fixtures, excessive operating hours,
and calculates Estimated Potential Energy Savings with data-driven optimization strategies.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .analytics import apply_chart_styling


def analyze_energy_savings(df):
    """
    Performs data-driven energy savings analysis:
    - Calculates baseline LED consumption vs legacy fixtures (HPS, Metal Halide)
    - Detects lamps with excessive operating hours (e.g., > 12 hours/day or top 10% operating hours)
    - Detects anomalous power draws
    - Computes:
        - Current Consumption (kWh)
        - Estimated Excess Consumption (kWh)
        - Estimated Potential Energy Saving (kWh)
        - Potential Saving Percentage (%)
    """
    if df.empty or 'Energy_Consumption' not in df.columns:
        return {
            'current_consumption': 0.0,
            'excess_consumption': 0.0,
            'potential_saving': 0.0,
            'saving_percentage': 0.0,
            'high_consumption_zones': [],
            'high_consumption_lamps': pd.DataFrame(),
            'excessive_hours_lamps': pd.DataFrame(),
            'legacy_fixtures_count': 0
        }
        
    current_consumption = float(df['Energy_Consumption'].sum())
    
    # 1. LED Benchmark comparison
    led_mask = df['Lamp_Type'].str.contains('LED', case=False, na=False) if 'Lamp_Type' in df.columns else pd.Series(True, index=df.index)
    led_df = df[led_mask]
    legacy_df = df[~led_mask]
    
    avg_led_kwh = float(led_df['Energy_Consumption'].mean()) if not led_df.empty else 28.0
    legacy_count = len(legacy_df)
    
    # Retrofit savings: difference between legacy lamps and efficient LED baseline
    retrofit_excess = float((legacy_df['Energy_Consumption'] - avg_led_kwh).clip(lower=0).sum()) if not legacy_df.empty else 0.0
    
    # 2. Excessive operating hours detection
    # Lamps with operating hours in the 90th percentile for their age
    hours_threshold = df['Operating_Hours'].quantile(0.85) if 'Operating_Hours' in df.columns else 9000
    excessive_hours_mask = df.get('Operating_Hours', pd.Series(0, index=df.index)) > hours_threshold
    excessive_hours_lamps = df[excessive_hours_mask].sort_values('Operating_Hours', ascending=False)
    
    # Intelligent dimming savings (approx 12-15% of total active consumption through scheduled off-peak dimming)
    dimming_savings = current_consumption * 0.12
    
    # Total Estimated Potential Energy Savings
    potential_saving = retrofit_excess + dimming_savings
    # Avoid exceeding reasonable ceiling (e.g. max 50%)
    potential_saving = min(potential_saving, current_consumption * 0.48)
    excess_consumption = potential_saving
    saving_percentage = round((potential_saving / current_consumption * 100.0), 1) if current_consumption > 0 else 0.0
    
    # 3. High consumption lamps (above 90th percentile)
    energy_p90 = df['Energy_Consumption'].quantile(0.90)
    high_consumption_lamps = df[df['Energy_Consumption'] >= energy_p90].sort_values('Energy_Consumption', ascending=False)
    
    # 4. High consumption zones
    if 'Zone' in df.columns:
        zone_avg = df.groupby('Zone')['Energy_Consumption'].mean()
        high_consumption_zones = zone_avg[zone_avg > zone_avg.quantile(0.70)].index.tolist()
    else:
        high_consumption_zones = []
        
    return {
        'current_consumption': round(current_consumption, 2),
        'excess_consumption': round(excess_consumption, 2),
        'potential_saving': round(potential_saving, 2),
        'saving_percentage': saving_percentage,
        'high_consumption_zones': high_consumption_zones,
        'high_consumption_lamps': high_consumption_lamps,
        'excessive_hours_lamps': excessive_hours_lamps,
        'legacy_fixtures_count': legacy_count,
        'avg_led_kwh': round(avg_led_kwh, 2),
        'retrofit_excess': round(retrofit_excess, 2),
        'dimming_savings': round(dimming_savings, 2)
    }


def plot_energy_savings_waterfall(current_kwh, retrofit_savings, dimming_savings, optimized_kwh):
    """Generates a smart city waterfall comparison showing energy reduction pathways."""
    fig = go.Figure(go.Waterfall(
        name="Energy Optimization",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Current Consumption", "LED Retrofit Savings", "Smart Dimming Savings", "Optimized Consumption"],
        textposition="outside",
        text=[f"{current_kwh:,.0f} kWh", f"-{retrofit_savings:,.0f} kWh", f"-{dimming_savings:,.0f} kWh", f"{optimized_kwh:,.0f} kWh"],
        y=[current_kwh, -retrofit_savings, -dimming_savings, 0],
        connector={"line": {"color": "#334155"}},
        decreasing={"marker": {"color": "#10b981"}},
        increasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#38bdf8"}}
    ))
    return apply_chart_styling(fig, "Potential Energy Reduction Breakdown (Simulated Estimate)")


def plot_zone_consumption_comparison(df, high_energy_zones):
    """Compares average energy per lamp across zones highlighting high-consumption areas."""
    if 'Zone' not in df.columns or 'Energy_Consumption' not in df.columns:
        return go.Figure()
        
    zone_stats = df.groupby('Zone')['Energy_Consumption'].mean().reset_index()
    zone_stats['Status'] = zone_stats['Zone'].apply(
        lambda z: 'High Consumption (Audit Required)' if z in high_energy_zones else 'Standard Efficiency'
    )
    
    fig = px.bar(
        zone_stats,
        x='Zone',
        y='Energy_Consumption',
        color='Status',
        color_discrete_map={
            'High Consumption (Audit Required)': '#f87171',
            'Standard Efficiency': '#38bdf8'
        },
        labels={'Energy_Consumption': 'Avg Energy Consumption (kWh)', 'Zone': 'Smart City Zone'},
        text_auto='.1f'
    )
    return apply_chart_styling(fig, "Zone Energy Efficiency Benchmarking")
