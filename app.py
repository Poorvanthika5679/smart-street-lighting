"""
Smart Street Lighting & Energy Usage Analytics System
Streamlit Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Smart Street Lighting Analytics",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from modules.preprocessing import (
    load_data,
    preprocess_data,
    get_dataset_summary,
    generate_sample_dataset
)
from modules.analytics import (
    calculate_dashboard_kpis,
    filter_dataset,
    get_energy_overview_stats,
    plot_energy_trend,
    plot_energy_by_zone,
    plot_energy_by_lamp_type,
    plot_operating_hours_by_zone,
    plot_lamp_status_distribution,
    plot_fault_distribution,
    get_zone_summary_table,
    get_single_zone_details
)
from modules.fault_prediction import (
    train_fault_prediction_model,
    predict_single_lamp,
    score_dataset_fault_risks
)
from modules.energy_saving import (
    analyze_energy_savings,
    plot_energy_savings_waterfall,
    plot_zone_consumption_comparison
)
from modules.recommendations import generate_recommendations

# ==========================================
# CUSTOM CSS INJECTION
# ==========================================
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline basic style if file not found
        st.markdown("""
        <style>
        .stApp { background-color: #0b1120; color: #f8fafc; }
        .kpi-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 1rem; }
        </style>
        """, unsafe_allow_html=True)

load_custom_css()

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'df_raw' not in st.session_state:
    sample_csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_street_lights.csv")
    # If sample CSV exists and has adequate records, load it; otherwise generate 650 records
    if os.path.exists(sample_csv_path):
        try:
            df_init = pd.read_csv(sample_csv_path)
            if len(df_init) < 500:
                df_init = generate_sample_dataset(n_records=650)
                df_init.to_csv(sample_csv_path, index=False)
        except Exception:
            df_init = generate_sample_dataset(n_records=650)
    else:
        df_init = generate_sample_dataset(n_records=650)
        os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)
        df_init.to_csv(sample_csv_path, index=False)
        
    st.session_state.df_raw = df_init
    st.session_state.is_sample_data = True
    st.session_state.dataset_source_label = "Demo / Sample Dataset (Municipal Simulation)"

if 'df_cleaned' not in st.session_state:
    df_clean, report = preprocess_data(st.session_state.df_raw)
    st.session_state.df_cleaned = df_clean
    st.session_state.cleaning_report = report

# Train ML model once or lazily
if 'ml_artifacts' not in st.session_state:
    pipeline, metrics, cm_fig, fi_fig, err = train_fault_prediction_model(st.session_state.df_cleaned)
    st.session_state.ml_pipeline = pipeline
    st.session_state.ml_metrics = metrics
    st.session_state.ml_cm_fig = cm_fig
    st.session_state.ml_fi_fig = fi_fig
    st.session_state.ml_error = err
    # Batch score dataframe
    st.session_state.df_scored = score_dataset_fault_risks(pipeline, st.session_state.df_cleaned)

# ==========================================
# SIDEBAR NAVIGATION & DATASET BADGE
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.8rem 0; text-align: center;">
        <span style="font-size: 2.2rem;">💡</span>
        <h2 style="margin: 0.2rem 0; font-size: 1.25rem; font-weight: 700; color: #38bdf8;">SMART LIGHTING AI</h2>
        <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">Smart City IoT & Energy Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset status indicator
    if st.session_state.get('is_sample_data', True):
        st.markdown("""
        <div style="margin-bottom: 1.2rem; text-align: center;">
            <span class="badge badge-demo">🏷️ Demo / Sample Dataset</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom: 1.2rem; text-align: center;">
            <span class="badge badge-low">✅ Uploaded Custom Dataset</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Navigation")
    pages = [
        "🏠 Dashboard",
        "📁 Data Upload & Preprocessing",
        "⚡ Energy Analytics",
        "🗺️ Zone Analysis",
        "🤖 Fault Prediction",
        "🚨 Fault Alerts",
        "🔧 Maintenance",
        "💡 Energy Saving",
        "📋 Recommendations"
    ]
    
    selected_page = st.radio("Go to Page", pages, label_visibility="collapsed")
    
    st.divider()
    st.markdown("""
    <div style="font-size: 0.72rem; color: #64748b; line-height: 1.4;">
        <b>Telemetry Status</b><br>
        Active Nodes: <b>{}</b><br>
        Monitored Zones: <b>{}</b><br>
        <i>Prototype v2.0 • SRS Compliant</i>
    </div>
    """.format(
        len(st.session_state.df_cleaned),
        st.session_state.df_cleaned['Zone'].nunique() if 'Zone' in st.session_state.df_cleaned.columns else 0
    ), unsafe_allow_html=True)

# Main Header Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">SMART STREET LIGHTING & ENERGY USAGE ANALYTICS SYSTEM</div>
    <div class="header-subtitle">AI-powered energy monitoring, fault prediction and maintenance decision support</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. MAIN DASHBOARD
# ==============================================================================
if selected_page == "🏠 Dashboard":
    df_data = st.session_state.df_scored
    kpis = calculate_dashboard_kpis(df_data)
    
    # Quick alert banner if sample data
    if st.session_state.get('is_sample_data', True):
        st.info("ℹ️ **Active Source**: Demo / Sample Dataset with 650+ simulated smart city street light records. You can upload your own municipal data in the **Data Upload & Preprocessing** tab.")

    # KPI Cards Row
    st.markdown("### 📊 Citywide Key Performance Indicators")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💡 Total Lamps</div>
            <div class="kpi-value">{kpis['total_lamps']:,}</div>
            <div class="kpi-subtext">Registered luminaires</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #34d399;">🟢 Active Lamps</div>
            <div class="kpi-value" style="color: #34d399;">{kpis['active_lamps']:,}</div>
            <div class="kpi-subtext">Currently illuminated</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #f87171;">🔴 Faulty Lamps</div>
            <div class="kpi-value" style="color: #f87171;">{kpis['faulty_lamps']:,}</div>
            <div class="kpi-subtext">Needs repair / offline</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⚡ Total Energy</div>
            <div class="kpi-value">{kpis['total_energy']:,.0f}</div>
            <div class="kpi-subtext">kWh total consumption</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📈 Avg Energy</div>
            <div class="kpi-value">{kpis['avg_energy']:.1f}</div>
            <div class="kpi-subtext">kWh/lamp monthly</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #fbbf24;">⚠️ High-Risk Lamps</div>
            <div class="kpi-value" style="color: #fbbf24;">{kpis['high_risk_lamps']:,}</div>
            <div class="kpi-subtext">Fault prob &ge; 70%</div>
        </div>
        """, unsafe_allow_html=True)
    with c7:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #38bdf8;">💰 Est. Savings</div>
            <div class="kpi-value" style="color: #38bdf8;">{kpis['potential_savings']:,.0f}</div>
            <div class="kpi-subtext">kWh potential saving</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Plotly Charts (2x2 Grid)
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        st.plotly_chart(plot_energy_trend(df_data), use_container_width=True)
        st.plotly_chart(plot_fault_distribution(df_data), use_container_width=True)
    with ch_col2:
        st.plotly_chart(plot_energy_by_zone(df_data), use_container_width=True)
        st.plotly_chart(plot_lamp_status_distribution(df_data), use_container_width=True)

    # Recent Fault Alerts & High-Risk Lamps Table
    st.markdown("### 🚨 Urgent Attention & High-Risk Luminaires")
    high_risk_df = df_data[df_data['Risk_Level'] == 'HIGH']
    
    if not high_risk_df.empty:
        display_cols = [
            'Lamp_ID', 'Zone', 'Lamp_Type', 'Fault_Probability', 'Risk_Level',
            'Energy_Consumption', 'Operating_Hours', 'Voltage', 'Recommended_Action'
        ]
        cols_to_show = [c for c in display_cols if c in high_risk_df.columns]
        st.dataframe(
            high_risk_df[cols_to_show].head(8).style.format({
                'Fault_Probability': '{:.1f}%',
                'Energy_Consumption': '{:.2f} kWh',
                'Operating_Hours': '{:,.0f} hrs',
                'Voltage': '{:.1f} V'
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("✅ No critical high-risk street lamps identified in the current dataset.")

    # Quick summary of the selected dataset
    with st.expander("📋 Quick Summary of Current Active Dataset"):
        st.write(f"- **Total Monitored Nodes**: {len(df_data)} lamps across {df_data['Zone'].nunique() if 'Zone' in df_data.columns else 'N/A'} administrative zones.")
        st.write(f"- **Fixture Composition**: {dict(df_data['Lamp_Type'].value_counts()) if 'Lamp_Type' in df_data.columns else 'N/A'}")
        st.write(f"- **Dataset Source**: {st.session_state.dataset_source_label}")
        st.write(f"- **Data Preprocessing Status**: Duplicates removed, telemetry values normalized, missing values imputed.")


# ==============================================================================
# 2. DATA UPLOAD & PREPROCESSING
# ==============================================================================
elif selected_page == "📁 Data Upload & Preprocessing":
    st.markdown("## 📁 Dataset Ingestion & Preprocessing Pipeline")
    st.markdown("Upload your smart street lighting dataset (CSV or Excel) or reset to the verified municipal demonstration dataset.")

    up_col1, up_col2 = st.columns([3, 1])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Street Light Telemetry Dataset (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Expected columns: Lamp_ID, Zone, Lamp_Type, Lamp_Status, Operating_Hours, Energy_Consumption, Date_Time, Voltage, Current, Previous_Faults, Lamp_Age, Fault_Status"
        )
    with up_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset to Demo Dataset", use_container_width=True):
            df_demo = generate_sample_dataset(n_records=650)
            st.session_state.df_raw = df_demo
            st.session_state.is_sample_data = True
            st.session_state.dataset_source_label = "Demo / Sample Dataset"
            df_cleaned, report = preprocess_data(df_demo)
            st.session_state.df_cleaned = df_cleaned
            st.session_state.cleaning_report = report
            # Re-train model
            pipeline, metrics, cm_fig, fi_fig, err = train_fault_prediction_model(df_cleaned)
            st.session_state.ml_pipeline = pipeline
            st.session_state.ml_metrics = metrics
            st.session_state.ml_cm_fig = cm_fig
            st.session_state.ml_fi_fig = fi_fig
            st.session_state.ml_error = err
            st.session_state.df_scored = score_dataset_fault_risks(pipeline, df_cleaned)
            st.success("Loaded realistic Demo Dataset with 650+ records.")
            st.rerun()

    if uploaded_file is not None:
        try:
            df_uploaded = load_data(uploaded_file)
            st.session_state.df_raw = df_uploaded
            st.session_state.is_sample_data = False
            st.session_state.dataset_source_label = f"Uploaded File: {uploaded_file.name}"
            
            # Preprocess
            df_cleaned, report = preprocess_data(df_uploaded)
            st.session_state.df_cleaned = df_cleaned
            st.session_state.cleaning_report = report
            
            # Re-train model
            pipeline, metrics, cm_fig, fi_fig, err = train_fault_prediction_model(df_cleaned)
            st.session_state.ml_pipeline = pipeline
            st.session_state.ml_metrics = metrics
            st.session_state.ml_cm_fig = cm_fig
            st.session_state.ml_fi_fig = fi_fig
            st.session_state.ml_error = err
            st.session_state.df_scored = score_dataset_fault_risks(pipeline, df_cleaned)
            st.success(f"Successfully processed `{uploaded_file.name}`!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # Preprocessing Pipeline Summary Cards
    st.markdown("### ⚙️ Preprocessing & Data Cleaning Summary")
    report = st.session_state.cleaning_report
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📥 Original Records</div>
            <div class="kpi-value">{report['original_records']:,}</div>
            <div class="kpi-subtext">Total ingested rows</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #fbbf24;">🔍 Missing Values</div>
            <div class="kpi-value" style="color: #fbbf24;">{report['missing_values_handled']:,}</div>
            <div class="kpi-subtext">Imputed (Median / Mode)</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #f87171;">✂️ Duplicate Records</div>
            <div class="kpi-value" style="color: #f87171;">{report['duplicates_removed']:,}</div>
            <div class="kpi-subtext">Detected and removed</div>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #34d399;">✨ Cleaned Records</div>
            <div class="kpi-value">{report['cleaned_records']:,}</div>
            <div class="kpi-subtext">Ready for analytics & ML</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Data Previews
    preview_tab1, preview_tab2 = st.tabs(["✨ Cleaned & Preprocessed Dataset", "📄 Raw Uploaded Dataset"])
    
    with preview_tab1:
        st.markdown(f"**Cleaned Dataset ({len(st.session_state.df_cleaned)} rows)**")
        st.dataframe(st.session_state.df_cleaned.head(25), use_container_width=True)
        
        # Download Cleaned Dataset
        csv_buffer = io.BytesIO()
        st.session_state.df_cleaned.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_buffer.getvalue(),
            file_name="cleaned_smart_street_lights.csv",
            mime="text/csv"
        )
        
    with preview_tab2:
        st.markdown(f"**Raw Ingested Dataset ({len(st.session_state.df_raw)} rows)**")
        st.dataframe(st.session_state.df_raw.head(25), use_container_width=True)


# ==============================================================================
# 3. ENERGY ANALYTICS
# ==============================================================================
elif selected_page == "⚡ Energy Analytics":
    st.markdown("## ⚡ Energy Usage & Consumption Analytics")
    st.markdown("Analyze power consumption patterns, filter by telemetry attributes, and inspect operational metrics.")
    
    df_clean = st.session_state.df_cleaned
    
    # Sidebar-style or Top Filters
    with st.expander("🔍 Interactive Data Filters", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            all_zones = sorted(df_clean['Zone'].unique()) if 'Zone' in df_clean.columns else []
            selected_zones = st.multiselect("Select Zones", options=all_zones, default=all_zones)
            
        with f_col2:
            all_types = sorted(df_clean['Lamp_Type'].unique()) if 'Lamp_Type' in df_clean.columns else []
            selected_types = st.multiselect("Select Lamp Types", options=all_types, default=all_types)
            
        with f_col3:
            all_statuses = sorted(df_clean['Lamp_Status'].unique()) if 'Lamp_Status' in df_clean.columns else []
            selected_statuses = st.multiselect("Select Lamp Status", options=all_statuses, default=all_statuses)
            
        with f_col4:
            if 'Date' in df_clean.columns and not df_clean['Date'].isnull().all():
                min_date = df_clean['Date'].min()
                max_date = df_clean['Date'].max()
                date_filter = st.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
            else:
                date_filter = None

    # Filter data dynamically
    df_filtered = filter_dataset(
        df_clean,
        date_range=date_filter if isinstance(date_filter, (list, tuple)) and len(date_filter) == 2 else None,
        zones=selected_zones,
        lamp_types=selected_types,
        statuses=selected_statuses
    )

    if df_filtered.empty:
        st.warning("⚠️ No records match the selected filter criteria. Please broaden your selection.")
    else:
        # Energy Overview Stats
        stats = get_energy_overview_stats(df_filtered)
        
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">⚡ Total Consumption</div>
                <div class="kpi-value">{stats['total']:,.1f}</div>
                <div class="kpi-subtext">kWh total energy</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📊 Average Consumption</div>
                <div class="kpi-value">{stats['avg']:.1f}</div>
                <div class="kpi-subtext">kWh per fixture</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📉 Minimum Draw</div>
                <div class="kpi-value">{stats['min']:.1f}</div>
                <div class="kpi-subtext">kWh lowest reading</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📈 Maximum Draw</div>
                <div class="kpi-value">{stats['max']:.1f}</div>
                <div class="kpi-subtext">kWh peak reading</div>
            </div>
            """, unsafe_allow_html=True)
        with s5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">⏱️ Total Operating Hours</div>
                <div class="kpi-value">{stats['total_hours']:,.0f}</div>
                <div class="kpi-subtext">Cumulative burn hours</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Charts
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.plotly_chart(plot_energy_trend(df_filtered), use_container_width=True)
            st.plotly_chart(plot_energy_by_lamp_type(df_filtered), use_container_width=True)
        with col_ch2:
            st.plotly_chart(plot_energy_by_zone(df_filtered), use_container_width=True)
            st.plotly_chart(plot_operating_hours_by_zone(df_filtered), use_container_width=True)


# ==============================================================================
# 4. ZONE ANALYSIS
# ==============================================================================
elif selected_page == "🗺️ Zone Analysis":
    st.markdown("## 🗺️ Smart City Zone Analysis")
    st.markdown("Evaluate zone-level energy efficiency benchmarks, identify high-wastage sectors, and drill into individual zones.")

    df_clean = st.session_state.df_cleaned
    zone_table, high_energy_zones = get_zone_summary_table(df_clean)

    if zone_table.empty:
        st.warning("No zone telemetry available.")
    else:
        # High Energy Wastage Banner
        if high_energy_zones:
            st.warning(f"⚠️ **High Energy Consumption Detected**: The following zones exceed statistical consumption benchmarks (>= 70th percentile of zone averages): **{', '.join(high_energy_zones)}**. Optimization audit recommended.")
        
        # Zone comparison table
        st.markdown("### 📊 Zone Comparison Matrix")
        st.dataframe(
            zone_table.style.format({
                'Total_Energy': '{:,.2f} kWh',
                'Avg_Energy': '{:.2f} kWh',
                'Total_Hours': '{:,.0f} hrs',
                'Fault_Rate': '{:.1f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Zone Drilldown
        st.markdown("### 🔍 Zone Telemetry Drilldown")
        all_zones = zone_table['Zone'].tolist()
        selected_zone = st.selectbox("Select Zone for Detailed Investigation", all_zones)

        if selected_zone:
            summary, top_lamps, type_counts = get_single_zone_details(df_clean, selected_zone)
            
            # Zone KPI cards
            z1, z2, z3, z4, z5 = st.columns(5)
            with z1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">💡 Luminaires</div>
                    <div class="kpi-value">{summary['total_lamps']:,}</div>
                    <div class="kpi-subtext">Lamps in {selected_zone}</div>
                </div>
                """, unsafe_allow_html=True)
            with z2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">⚡ Total Energy</div>
                    <div class="kpi-value">{summary['total_energy']:,.1f}</div>
                    <div class="kpi-subtext">kWh total consumption</div>
                </div>
                """, unsafe_allow_html=True)
            with z3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">📈 Avg Energy</div>
                    <div class="kpi-value">{summary['avg_energy']:.1f}</div>
                    <div class="kpi-subtext">kWh per fixture</div>
                </div>
                """, unsafe_allow_html=True)
            with z4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">⏱️ Total Hours</div>
                    <div class="kpi-value">{summary['total_hours']:,.0f}</div>
                    <div class="kpi-subtext">Operating hours</div>
                </div>
                """, unsafe_allow_html=True)
            with z5:
                rate_color = "#f87171" if summary['fault_rate'] > 15.0 else "#34d399"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label" style="color: {rate_color};">🔴 Fault Rate</div>
                    <div class="kpi-value" style="color: {rate_color};">{summary['fault_rate']:.1f}%</div>
                    <div class="kpi-subtext">{summary['fault_count']} faulty fixtures</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            z_col1, z_col2 = st.columns([1, 1.4])
            with z_col1:
                # Lamp type distribution pie
                fig_zone_type = plot_zone_consumption_comparison(df_clean, high_energy_zones)
                st.plotly_chart(fig_zone_type, use_container_width=True)
                
            with z_col2:
                st.markdown(f"**Top High-Consumption Fixtures in {selected_zone}**")
                st.dataframe(
                    top_lamps.style.format({
                        'Energy_Consumption': '{:.2f} kWh',
                        'Operating_Hours': '{:,.0f} hrs',
                        'Voltage': '{:.1f} V',
                        'Current': '{:.2f} A'
                    }),
                    use_container_width=True,
                    hide_index=True
                )


# ==============================================================================
# 5. MACHINE LEARNING FAULT PREDICTION
# ==============================================================================
elif selected_page == "🤖 Fault Prediction":
    st.markdown("## 🤖 Machine Learning Fault Prediction")
    st.markdown("Supervised Random Forest Classifier for predictive maintenance and real-time failure probability estimation.")

    df_clean = st.session_state.df_cleaned
    
    # Check if Fault_Status exists
    if 'Fault_Status' not in df_clean.columns or st.session_state.ml_error:
        err_msg = st.session_state.ml_error if st.session_state.ml_error else "Fault prediction requires labeled Fault_Status data."
        st.warning(f"⚠️ **{err_msg}**\n\nThe application will continue operating with other energy analytics modules.")
    else:
        metrics = st.session_state.ml_metrics
        pipeline = st.session_state.ml_pipeline
        
        # Model Performance Cards
        st.markdown("### 🎯 Model Evaluation Metrics (Random Forest Classifier)")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label" style="color: #38bdf8;">🎯 Accuracy</div>
                <div class="kpi-value" style="color: #38bdf8;">{metrics['accuracy']:.1f}%</div>
                <div class="kpi-subtext">Overall correct rate</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🔍 Precision</div>
                <div class="kpi-value">{metrics['precision']:.1f}%</div>
                <div class="kpi-subtext">True positive accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📡 Recall</div>
                <div class="kpi-value">{metrics['recall']:.1f}%</div>
                <div class="kpi-subtext">Fault detection sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">⚖️ F1-Score</div>
                <div class="kpi-value">{metrics['f1']:.1f}%</div>
                <div class="kpi-subtext">Harmonic mean balance</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🧪 Test Split</div>
                <div class="kpi-value">{metrics['test_samples']}</div>
                <div class="kpi-subtext">Samples evaluated</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion Matrix & Feature Importances
        col_cm, col_fi = st.columns(2)
        with col_cm:
            if st.session_state.ml_cm_fig:
                st.plotly_chart(st.session_state.ml_cm_fig, use_container_width=True)
        with col_fi:
            if st.session_state.ml_fi_fig:
                st.plotly_chart(st.session_state.ml_fi_fig, use_container_width=True)

        st.divider()

        # Individual Lamp Prediction Section
        st.markdown("### 🔬 Individual Lamp Fault Risk Scorer")
        st.markdown("Select a specific luminaire from the active grid to view electrical telemetry and run the ML risk prediction engine.")

        lamp_ids = df_clean['Lamp_ID'].unique().tolist()
        
        # Interactive selector
        sel_c1, sel_c2 = st.columns([1.2, 2])
        with sel_c1:
            selected_lamp_id = st.selectbox("Select Lamp ID", lamp_ids)
            lamp_row = df_clean[df_clean['Lamp_ID'] == selected_lamp_id].iloc[0]
            
            predict_btn = st.button("🚀 PREDICT FAULT RISK", use_container_width=True, type="primary")

        with sel_c2:
            st.markdown(f"**Telemetry Profile: `{selected_lamp_id}`**")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.write(f"• **Zone**: {lamp_row.get('Zone', 'N/A')}")
                st.write(f"• **Lamp Type**: {lamp_row.get('Lamp_Type', 'N/A')}")
                st.write(f"• **Status**: {lamp_row.get('Lamp_Status', 'N/A')}")
            with p_col2:
                st.write(f"• **Energy Draw**: {lamp_row.get('Energy_Consumption', 0):.2f} kWh")
                st.write(f"• **Operating Hours**: {lamp_row.get('Operating_Hours', 0):,.0f} hrs")
                st.write(f"• **Lamp Age**: {lamp_row.get('Lamp_Age', 0)} months")
            with p_col3:
                st.write(f"• **Voltage**: {lamp_row.get('Voltage', 0):.1f} V")
                st.write(f"• **Current**: {lamp_row.get('Current', 0):.2f} A")
                st.write(f"• **Previous Faults**: {lamp_row.get('Previous_Faults', 0)}")

        if predict_btn:
            prob, risk_level, risk_color = predict_single_lamp(pipeline, lamp_row)
            
            st.markdown("<br>", unsafe_allow_html=True)
            res_c1, res_c2 = st.columns([1.5, 2.5])
            with res_c1:
                st.markdown(f"""
                <div style="background: #1e293b; border: 2px solid {risk_color}; border-radius: 12px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Predicted Fault Risk</div>
                    <div style="font-size: 2.8rem; font-weight: 800; color: {risk_color}; margin: 0.5rem 0;">{prob:.1f}%</div>
                    <div><span class="badge" style="background: {risk_color}22; color: {risk_color}; border: 1px solid {risk_color}55; font-size: 1rem; padding: 0.4rem 1.2rem;">{risk_level} RISK</span></div>
                </div>
                """, unsafe_allow_html=True)
            with res_c2:
                st.markdown("#### Diagnostic Insights")
                if risk_level == "HIGH":
                    st.error(f"⚠️ **High Probability of Imminent Malfunction**: Fixture {selected_lamp_id} shows electrical strain or extreme operational wear. Priority work order recommended.")
                elif risk_level == "MEDIUM":
                    st.warning(f"⚡ **Moderate Wear Detected**: Fixture {selected_lamp_id} displays borderline operational metrics. Recommended for condition monitoring.")
                else:
                    st.success(f"✅ **Optimal Operational Health**: Telemetry values for {selected_lamp_id} fall well within safe operating parameters.")

                st.markdown("""
                <div class="disclaimer-box">
                    ℹ️ <b>Decision-Support Notice</b>: Predicted Fault Risk is an algorithmic estimate calculated from historical electrical telemetry and usage hours. It does not guarantee an actual hardware failure.
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# 6. FAULT ALERTS
# ==============================================================================
elif selected_page == "🚨 Fault Alerts":
    st.markdown("## 🚨 Smart City Fault Alerts & Priority Watchlist")
    st.markdown("Real-time watchlist of smart street lights requiring supervisory intervention, filtered by operational criticality.")

    df_scored = st.session_state.df_scored

    # Filter Bar
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        zones_avail = ['All Zones'] + sorted(df_scored['Zone'].unique().tolist())
        filter_zone = st.selectbox("Filter by Zone", zones_avail)
    with f_c2:
        risk_avail = ['All Risk Levels', 'HIGH', 'MEDIUM', 'LOW']
        filter_risk = st.selectbox("Filter by Risk Level", risk_avail)
    with f_c3:
        types_avail = ['All Types'] + sorted(df_scored['Lamp_Type'].unique().tolist())
        filter_type = st.selectbox("Filter by Lamp Type", types_avail)

    # Filter logic
    df_alerts = df_scored.copy()
    if filter_zone != 'All Zones':
        df_alerts = df_alerts[df_alerts['Zone'] == filter_zone]
    if filter_risk != 'All Risk Levels':
        df_alerts = df_alerts[df_alerts['Risk_Level'] == filter_risk]
    if filter_type != 'All Types':
        df_alerts = df_alerts[df_alerts['Lamp_Type'] == filter_type]

    # Metrics on filtered alerts
    high_cnt = (df_alerts['Risk_Level'] == 'HIGH').sum()
    med_cnt = (df_alerts['Risk_Level'] == 'MEDIUM').sum()
    low_cnt = (df_alerts['Risk_Level'] == 'LOW').sum()

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📋 Filtered Items</div>
            <div class="kpi-value">{len(df_alerts):,}</div>
            <div class="kpi-subtext">Total lamps in view</div>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #f87171;">🔴 High Risk</div>
            <div class="kpi-value" style="color: #f87171;">{high_cnt:,}</div>
            <div class="kpi-subtext">Prob &ge; 70%</div>
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #fbbf24;">🟡 Medium Risk</div>
            <div class="kpi-value" style="color: #fbbf24;">{med_cnt:,}</div>
            <div class="kpi-subtext">Prob 35% - 70%</div>
        </div>
        """, unsafe_allow_html=True)
    with a4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #34d399;">🟢 Low Risk</div>
            <div class="kpi-value" style="color: #34d399;">{low_cnt:,}</div>
            <div class="kpi-subtext">Prob &lt; 35%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alerts Table
    if df_alerts.empty:
        st.info("No street lamps match the selected alert filter criteria.")
    else:
        table_cols = [
            'Lamp_ID', 'Zone', 'Lamp_Type', 'Fault_Probability', 'Risk_Level',
            'Energy_Consumption', 'Operating_Hours', 'Recommended_Action'
        ]
        cols_present = [c for c in table_cols if c in df_alerts.columns]
        
        # Sort by Fault_Probability descending
        df_display = df_alerts.sort_values('Fault_Probability', ascending=False)[cols_present]
        
        st.dataframe(
            df_display.style.format({
                'Fault_Probability': '{:.1f}%',
                'Energy_Consumption': '{:.2f} kWh',
                'Operating_Hours': '{:,.0f} hrs'
            }),
            use_container_width=True,
            hide_index=True
        )


# ==============================================================================
# 7. MAINTENANCE DECISION SUPPORT
# ==============================================================================
elif selected_page == "🔧 Maintenance":
    st.markdown("## 🔧 Maintenance Decision Support System")
    st.markdown("Automated prioritization engine ranking street lighting assets for maintenance dispatch based on composite multi-factor risk scores.")

    df_scored = st.session_state.df_scored

    # Compute composite maintenance priority score:
    # 50% Fault Probability + 20% Operating Hours ratio + 20% Previous Faults + 10% Energy Draw
    max_hours = df_scored['Operating_Hours'].max() if df_scored['Operating_Hours'].max() > 0 else 1
    max_energy = df_scored['Energy_Consumption'].max() if df_scored['Energy_Consumption'].max() > 0 else 1
    
    df_maint = df_scored.copy()
    priority_score = (
        df_maint['Fault_Probability'] * 0.50 +
        (df_maint['Operating_Hours'] / max_hours * 100) * 0.20 +
        (df_maint['Previous_Faults'] / 5.0 * 100).clip(upper=100) * 0.20 +
        (df_maint['Energy_Consumption'] / max_energy * 100) * 0.10
    )
    df_maint['Maintenance_Priority_Score'] = priority_score.round(1)
    
    def classify_priority(score):
        if score >= 65.0:
            return 'P1 - Immediate Dispatch'
        elif score >= 45.0:
            return 'P2 - High Priority'
        elif score >= 30.0:
            return 'P3 - Scheduled Review'
        return 'P4 - Routine Cycle'
        
    df_maint['Maintenance_Priority'] = df_maint['Maintenance_Priority_Score'].apply(classify_priority)

    # Summary cards
    p1_count = (df_maint['Maintenance_Priority'] == 'P1 - Immediate Dispatch').sum()
    p2_count = (df_maint['Maintenance_Priority'] == 'P2 - High Priority').sum()
    p3_count = (df_maint['Maintenance_Priority'] == 'P3 - Scheduled Review').sum()

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #ef4444;">🚨 P1 Immediate Dispatch</div>
            <div class="kpi-value" style="color: #ef4444;">{p1_count}</div>
            <div class="kpi-subtext">Immediate physical inspection</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #f59e0b;">⚠️ P2 High Priority</div>
            <div class="kpi-value" style="color: #f59e0b;">{p2_count}</div>
            <div class="kpi-subtext">Preventive maintenance batch</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #38bdf8;">📋 P3 Scheduled Review</div>
            <div class="kpi-value" style="color: #38bdf8;">{p3_count}</div>
            <div class="kpi-subtext">Condition monitoring</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📋 Prioritized Work Order Recommendations")
    
    maint_cols = [
        'Lamp_ID', 'Zone', 'Risk_Level', 'Fault_Probability', 'Energy_Consumption',
        'Operating_Hours', 'Previous_Faults', 'Maintenance_Priority', 'Recommended_Action'
    ]
    cols_to_render = [c for c in maint_cols if c in df_maint.columns]
    
    df_sorted_maint = df_maint.sort_values('Maintenance_Priority_Score', ascending=False)[cols_to_render]
    
    st.dataframe(
        df_sorted_maint.head(50).style.format({
            'Fault_Probability': '{:.1f}%',
            'Energy_Consumption': '{:.2f} kWh',
            'Operating_Hours': '{:,.0f} hrs'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="disclaimer-box">
        💡 <b>Decision-Support Disclaimer</b>: The maintenance priorities generated here are advisory recommendations calculated by the analytics engine to assist smart-city operations teams. They do not constitute binding automated maintenance orders.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 8. ENERGY SAVING
# ==============================================================================
elif selected_page == "💡 Energy Saving":
    st.markdown("## 💡 Municipal Energy Saving & Wastage Analysis")
    st.markdown("Detect high-energy wastage pockets, evaluate LED conversion benefits, and view estimated potential energy savings.")

    df_clean = st.session_state.df_cleaned
    savings = analyze_energy_savings(df_clean)

    # 4 Key metrics
    es1, es2, es3, es4 = st.columns(4)
    with es1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⚡ Current Consumption</div>
            <div class="kpi-value">{savings['current_consumption']:,.0f}</div>
            <div class="kpi-subtext">kWh total monthly baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with es2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #f87171;">⚠️ Est. Excess Draw</div>
            <div class="kpi-value" style="color: #f87171;">{savings['excess_consumption']:,.0f}</div>
            <div class="kpi-subtext">kWh avoidable energy usage</div>
        </div>
        """, unsafe_allow_html=True)
    with es3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #34d399;">💰 Potential Savings</div>
            <div class="kpi-value" style="color: #34d399;">{savings['potential_saving']:,.0f}</div>
            <div class="kpi-subtext">kWh monthly reduction</div>
        </div>
        """, unsafe_allow_html=True)
    with es4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label" style="color: #38bdf8;">📉 Saving Percentage</div>
            <div class="kpi-value" style="color: #38bdf8;">{savings['saving_percentage']:.1f}%</div>
            <div class="kpi-subtext">Citywide consumption cut</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Energy reduction waterfall & Zone comparison
    opt_kwh = max(0.0, savings['current_consumption'] - savings['potential_saving'])
    wf_fig = plot_energy_savings_waterfall(
        savings['current_consumption'],
        savings['retrofit_excess'],
        savings['dimming_savings'],
        opt_kwh
    )
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.plotly_chart(wf_fig, use_container_width=True)
    with col_w2:
        zone_eff_fig = plot_zone_consumption_comparison(df_clean, savings['high_consumption_zones'])
        st.plotly_chart(zone_eff_fig, use_container_width=True)

    # High-Consumption Fixtures Table
    st.markdown("### 🔍 Highest Energy-Consuming Luminaires (Audit Targets)")
    if not savings['high_consumption_lamps'].empty:
        top_draw_cols = ['Lamp_ID', 'Zone', 'Lamp_Type', 'Energy_Consumption', 'Operating_Hours', 'Voltage', 'Current']
        st.dataframe(
            savings['high_consumption_lamps'][top_draw_cols].head(10).style.format({
                'Energy_Consumption': '{:.2f} kWh',
                'Operating_Hours': '{:,.0f} hrs',
                'Voltage': '{:.1f} V',
                'Current': '{:.2f} A'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("""
    <div class="disclaimer-box">
        💡 <b>Important Notice</b>: The values displayed above represent <b>Estimated Potential Energy Saving</b> derived from statistical regression against modern LED and smart dimming profiles. They do not represent guaranteed real-world financial or wattage savings.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 9. RECOMMENDATIONS
# ==============================================================================
elif selected_page == "📋 Recommendations":
    st.markdown("## 📋 Smart City Actionable Recommendations")
    st.markdown("Data-driven decision support for municipal authorities, categorized into maintenance interventions and energy efficiency optimizations.")

    df_scored = st.session_state.df_scored
    savings = analyze_energy_savings(df_scored)
    recs = generate_recommendations(st.session_state.df_cleaned, df_scored, savings)

    # Category A: Maintenance Recommendations
    st.markdown("### 🔧 Category A: Maintenance Recommendations")
    for rec in recs['maintenance_recs']:
        priority_class = "rec-card-urgent" if rec['priority'] == 'URGENT' else "rec-card-warning"
        badge_color = "#f87171" if rec['priority'] == 'URGENT' else "#fbbf24"
        st.markdown(f"""
        <div class="rec-card {priority_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <b style="font-size: 1.05rem; color: #f8fafc;">{rec['title']}</b>
                <span class="badge" style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}44;">{rec['priority']}</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">{rec['description']}</p>
            <div style="font-size: 0.82rem; color: #38bdf8;"><b>Recommended Intervention:</b> {rec['action']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Category B: Energy Optimization Recommendations
    st.markdown("### ⚡ Category B: Energy Optimization Recommendations")
    for rec in recs['energy_recs']:
        badge_color = "#38bdf8" if rec['priority'] == 'HIGH' else "#34d399"
        st.markdown(f"""
        <div class="rec-card rec-card-info">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <b style="font-size: 1.05rem; color: #f8fafc;">{rec['title']}</b>
                <span class="badge" style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}44;">{rec['priority']} IMPACT</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem;">{rec['description']}</p>
            <div style="display: flex; justify-content: space-between; font-size: 0.82rem;">
                <span style="color: #34d399;"><b>Estimated Impact:</b> {rec['kwh_impact']}</span>
                <span style="color: #38bdf8;"><b>Action:</b> {rec['action']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
