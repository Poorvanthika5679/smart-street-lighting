"""
Analytics Module
Provides calculation routines and interactive Plotly visualizations for:
- Main Dashboard KPIs & trends
- Energy Analytics (time series, distributions, breakdown by zone & lamp type)
- Zone-level Analytics (benchmarks, high-wastage identification, zone drilldown)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Smart city chart color palette
CHART_THEME = {
    'bg': '#0f172a',
    'paper': '#1e293b',
    'text': '#f8fafc',
    'grid': '#334155',
    'primary': '#38bdf8',
    'secondary': '#818cf8',
    'accent': '#34d399',
    'warning': '#fbbf24',
    'danger': '#f87171',
    'colors': ['#38bdf8', '#818cf8', '#34d399', '#fbbf24', '#f87171', '#c084fc']
}


def apply_chart_styling(fig, title=""):
    """Applies a consistent dark navy smart-city aesthetic to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, sans-serif", size=15, color="#f8fafc"),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor=CHART_THEME['paper'],
        plot_bgcolor=CHART_THEME['bg'],
        font=dict(family="Inter, sans-serif", color=CHART_THEME['text'], size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=CHART_THEME['grid'],
            zeroline=False,
            showline=True,
            linecolor=CHART_THEME['grid']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=CHART_THEME['grid'],
            zeroline=False,
            showline=True,
            linecolor=CHART_THEME['grid']
        ),
        hoverlabel=dict(
            bgcolor="#0b1120",
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    return fig


def calculate_dashboard_kpis(df):
    """
    Computes summary KPI indicators for the main dashboard:
    - Total Lamps
    - Active Lamps
    - Faulty Lamps
    - Total Energy Consumption (kWh)
    - Average Energy Consumption (kWh)
    - High-Risk Lamps
    - Estimated Potential Energy Savings (kWh)
    """
    if df.empty:
        return {
            'total_lamps': 0,
            'active_lamps': 0,
            'faulty_lamps': 0,
            'total_energy': 0.0,
            'avg_energy': 0.0,
            'high_risk_lamps': 0,
            'potential_savings': 0.0
        }
        
    total_lamps = len(df)
    active_lamps = int((df['Lamp_Status'].str.lower() == 'active').sum()) if 'Lamp_Status' in df.columns else total_lamps
    
    # Faulty lamps: either marked 'Maintenance Required' or labeled Fault_Status == 1
    if 'Fault_Status' in df.columns:
        faulty_lamps = int((df['Fault_Status'] == 1).sum())
    elif 'Lamp_Status' in df.columns:
        faulty_lamps = int(df['Lamp_Status'].str.lower().str.contains('maintenance|fault|error').sum())
    else:
        faulty_lamps = 0
        
    total_energy = float(df['Energy_Consumption'].sum()) if 'Energy_Consumption' in df.columns else 0.0
    avg_energy = float(df['Energy_Consumption'].mean()) if 'Energy_Consumption' in df.columns else 0.0
    
    # High-risk lamps: voltage anomaly, past faults >= 3, or high predicted probability if scored
    if 'Risk_Level' in df.columns:
        high_risk_lamps = int((df['Risk_Level'] == 'HIGH').sum())
    else:
        # Fallback heuristic if not yet run through ML
        high_risk_condition = (
            (df.get('Voltage', pd.Series(220, index=df.index)) < 195) |
            (df.get('Voltage', pd.Series(220, index=df.index)) > 245) |
            (df.get('Previous_Faults', pd.Series(0, index=df.index)) >= 3)
        )
        high_risk_lamps = int(high_risk_condition.sum())
        
    # Estimated potential energy savings (approx. 18-28% through LED retrofit & intelligent dimming)
    # Calculated data-driven: difference between legacy lamps (HPS/MH) and smart LED baseline
    led_avg = df[df['Lamp_Type'].str.contains('LED', case=False, na=False)]['Energy_Consumption'].mean() if 'Lamp_Type' in df.columns else avg_energy * 0.5
    if pd.isna(led_avg) or led_avg <= 0:
        led_avg = avg_energy * 0.5
        
    excess = 0.0
    if 'Lamp_Type' in df.columns and 'Energy_Consumption' in df.columns:
        legacy_lamps = df[~df['Lamp_Type'].str.contains('LED', case=False, na=False)]
        excess = float((legacy_lamps['Energy_Consumption'] - led_avg).clip(lower=0).sum())
        # Add 12% dimming savings on active LEDs
        led_savings = float(df[df['Lamp_Type'].str.contains('LED', case=False, na=False)]['Energy_Consumption'].sum() * 0.12)
        potential_savings = excess + led_savings
    else:
        potential_savings = total_energy * 0.22
        
    return {
        'total_lamps': total_lamps,
        'active_lamps': active_lamps,
        'faulty_lamps': faulty_lamps,
        'total_energy': round(total_energy, 2),
        'avg_energy': round(avg_energy, 2),
        'high_risk_lamps': high_risk_lamps,
        'potential_savings': round(potential_savings, 2)
    }


def filter_dataset(df, date_range=None, zones=None, lamp_types=None, statuses=None):
    """
    Applies multi-attribute dynamic filters to dataset.
    """
    filtered = df.copy()
    
    if date_range and len(date_range) == 2 and 'Date' in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[(filtered['Date'] >= start_date) & (filtered['Date'] <= end_date)]
        
    if zones and len(zones) > 0 and 'Zone' in filtered.columns:
        filtered = filtered[filtered['Zone'].isin(zones)]
        
    if lamp_types and len(lamp_types) > 0 and 'Lamp_Type' in filtered.columns:
        filtered = filtered[filtered['Lamp_Type'].isin(lamp_types)]
        
    if statuses and len(statuses) > 0 and 'Lamp_Status' in filtered.columns:
        filtered = filtered[filtered['Lamp_Status'].isin(statuses)]
        
    return filtered


def get_energy_overview_stats(df):
    """Returns basic min, max, avg, total energy, and total operating hours."""
    if df.empty or 'Energy_Consumption' not in df.columns:
        return {'total': 0.0, 'avg': 0.0, 'min': 0.0, 'max': 0.0, 'total_hours': 0.0}
        
    return {
        'total': round(float(df['Energy_Consumption'].sum()), 2),
        'avg': round(float(df['Energy_Consumption'].mean()), 2),
        'min': round(float(df['Energy_Consumption'].min()), 2),
        'max': round(float(df['Energy_Consumption'].max()), 2),
        'total_hours': round(float(df['Operating_Hours'].sum()), 1) if 'Operating_Hours' in df.columns else 0.0
    }


# ==========================================
# CHART GENERATORS (Plotly)
# ==========================================

def plot_energy_trend(df):
    """Energy consumption aggregated over time."""
    if 'Date' in df.columns and 'Energy_Consumption' in df.columns:
        trend = df.groupby('Date')['Energy_Consumption'].sum().reset_index().sort_values('Date')
        fig = px.area(
            trend,
            x='Date',
            y='Energy_Consumption',
            labels={'Energy_Consumption': 'Total Energy (kWh)', 'Date': 'Date'},
            color_discrete_sequence=['#38bdf8']
        )
        fig.update_traces(line=dict(width=2.5), fillcolor='rgba(56, 189, 248, 0.15)')
    else:
        fig = go.Figure()
        
    return apply_chart_styling(fig, "Energy Consumption Trend Over Time")


def plot_energy_by_zone(df):
    """Total and average energy consumption by zone."""
    if 'Zone' in df.columns and 'Energy_Consumption' in df.columns:
        zone_data = df.groupby('Zone').agg(
            Total_Energy=('Energy_Consumption', 'sum'),
            Avg_Energy=('Energy_Consumption', 'mean')
        ).reset_index().sort_values('Total_Energy', ascending=False)
        
        fig = px.bar(
            zone_data,
            x='Zone',
            y='Total_Energy',
            color='Avg_Energy',
            color_continuous_scale=['#38bdf8', '#818cf8', '#f87171'],
            labels={'Total_Energy': 'Total Energy (kWh)', 'Avg_Energy': 'Avg kWh/Lamp'},
            text_auto='.1f'
        )
    else:
        fig = go.Figure()
        
    return apply_chart_styling(fig, "Total Energy Consumption by Zone")


def plot_energy_by_lamp_type(df):
    """Energy distribution across lamp types."""
    if 'Lamp_Type' in df.columns and 'Energy_Consumption' in df.columns:
        fig = px.box(
            df,
            x='Lamp_Type',
            y='Energy_Consumption',
            color='Lamp_Type',
            color_discrete_sequence=CHART_THEME['colors'],
            labels={'Energy_Consumption': 'Energy Consumption (kWh)', 'Lamp_Type': 'Lamp Technology'}
        )
    else:
        fig = go.Figure()
        
    return apply_chart_styling(fig, "Energy Consumption Distribution by Lamp Type")


def plot_operating_hours_by_zone(df):
    """Total operating hours per zone."""
    if 'Zone' in df.columns and 'Operating_Hours' in df.columns:
        hours_data = df.groupby('Zone')['Operating_Hours'].sum().reset_index().sort_values('Operating_Hours', ascending=False)
        fig = px.bar(
            hours_data,
            x='Zone',
            y='Operating_Hours',
            color_discrete_sequence=['#818cf8'],
            labels={'Operating_Hours': 'Operating Hours (hrs)', 'Zone': 'Smart City Zone'},
            text_auto='.0f'
        )
    else:
        fig = go.Figure()
        
    return apply_chart_styling(fig, "Total Operating Hours by Zone")


def plot_lamp_status_distribution(df):
    """Donut chart of lamp status (Active, Inactive, Maintenance Required)."""
    if 'Lamp_Status' in df.columns:
        status_counts = df['Lamp_Status'].value_counts().reset_index()
        status_counts.columns = ['Lamp_Status', 'Count']
        
        color_map = {
            'Active': '#10b981',
            'Inactive': '#64748b',
            'Maintenance Required': '#ef4444'
        }
        
        fig = px.pie(
            status_counts,
            names='Lamp_Status',
            values='Count',
            hole=0.55,
            color='Lamp_Status',
            color_discrete_map=color_map
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
    else:
        fig = go.Figure()
        
    return apply_chart_styling(fig, "Lamp Status Distribution")


def plot_fault_distribution(df):
    """Faults grouped by zone and lamp type."""
    if 'Fault_Status' in df.columns and 'Zone' in df.columns:
        fault_df = df[df['Fault_Status'] == 1]
        if not fault_df.empty:
            fault_counts = fault_df.groupby(['Zone', 'Lamp_Type']).size().reset_index(name='Fault_Count')
            fig = px.bar(
                fault_counts,
                x='Zone',
                y='Fault_Count',
                color='Lamp_Type',
                barmode='stack',
                color_discrete_sequence=CHART_THEME['colors'],
                labels={'Fault_Count': 'Number of Faults', 'Zone': 'Zone'}
            )
        else:
            fig = go.Figure()
    else:
        fig = go.Figure()
        
    return apply_chart_styling(fig, "Fault Count by Zone & Lamp Technology")


# ==========================================
# ZONE ANALYSIS
# ==========================================

def get_zone_summary_table(df):
    """
    Computes per-zone comprehensive analytics:
    - Number of lamps
    - Total energy consumption (kWh)
    - Average energy consumption (kWh)
    - Operating hours
    - Fault count
    - Fault rate (%)
    
    Identifies zones with unusually high energy consumption using data-driven comparison
    (Zone average consumption > 75th percentile of all zone averages).
    """
    if df.empty or 'Zone' not in df.columns:
        return pd.DataFrame(), []
        
    grouped = df.groupby('Zone').agg(
        Total_Lamps=('Lamp_ID', 'count'),
        Total_Energy=('Energy_Consumption', 'sum'),
        Avg_Energy=('Energy_Consumption', 'mean'),
        Total_Hours=('Operating_Hours', 'sum'),
        Fault_Count=('Fault_Status', lambda s: int((s == 1).sum()) if 'Fault_Status' in df.columns else 0)
    ).reset_index()
    
    grouped['Fault_Rate'] = (grouped['Fault_Count'] / grouped['Total_Lamps'] * 100).round(1)
    grouped['Total_Energy'] = grouped['Total_Energy'].round(2)
    grouped['Avg_Energy'] = grouped['Avg_Energy'].round(2)
    grouped['Total_Hours'] = grouped['Total_Hours'].round(1)
    
    # Data-driven identification of high energy zones:
    # 75th percentile of zone averages or 1.15x the median zone average
    threshold_p75 = grouped['Avg_Energy'].quantile(0.70)
    city_avg = grouped['Avg_Energy'].mean()
    
    high_energy_zones = grouped[grouped['Avg_Energy'] >= threshold_p75]['Zone'].tolist()
    
    grouped['High_Energy_Flag'] = grouped['Zone'].apply(
        lambda z: '⚠️ High Consumption' if z in high_energy_zones else '✅ Normal'
    )
    
    return grouped.sort_values('Total_Energy', ascending=False), high_energy_zones


def get_single_zone_details(df, selected_zone):
    """Extracts detailed metrics and top high-consumption lamps for a specific zone."""
    zone_df = df[df['Zone'] == selected_zone]
    if zone_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
        
    total_lamps = len(zone_df)
    total_energy = round(float(zone_df['Energy_Consumption'].sum()), 2)
    avg_energy = round(float(zone_df['Energy_Consumption'].mean()), 2)
    total_hours = round(float(zone_df['Operating_Hours'].sum()), 1)
    fault_count = int((zone_df['Fault_Status'] == 1).sum()) if 'Fault_Status' in zone_df.columns else 0
    fault_rate = round(fault_count / total_lamps * 100, 1) if total_lamps > 0 else 0.0
    
    summary = {
        'total_lamps': total_lamps,
        'total_energy': total_energy,
        'avg_energy': avg_energy,
        'total_hours': total_hours,
        'fault_count': fault_count,
        'fault_rate': fault_rate
    }
    
    # Top 10 high-consumption lamps in zone
    top_lamps = zone_df.sort_values('Energy_Consumption', ascending=False).head(10)[[
        'Lamp_ID', 'Lamp_Type', 'Lamp_Status', 'Energy_Consumption', 'Operating_Hours', 'Voltage', 'Current'
    ]]
    
    # Lamp type breakdown
    type_counts = zone_df['Lamp_Type'].value_counts().reset_index()
    type_counts.columns = ['Lamp_Type', 'Count']
    
    return summary, top_lamps, type_counts
