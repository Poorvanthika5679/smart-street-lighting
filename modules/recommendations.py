"""
Recommendations Module
Generates data-driven decision-support recommendations:
- Category A: Maintenance Recommendations (fault prevention, priority inspections, overhauls)
- Category B: Energy Optimization Recommendations (retrofit, dimming, schedule optimization)
"""

import pandas as pd
import numpy as np


def generate_recommendations(df, scored_df=None, energy_analysis=None):
    """
    Analyzes the dataset and produces structured recommendations:
    Returns dict:
    {
        'maintenance_recs': [
            {'priority': 'URGENT' | 'HIGH' | 'MEDIUM', 'title': ..., 'description': ..., 'affected_count': ..., 'action': ...}
        ],
        'energy_recs': [
            {'priority': 'HIGH' | 'MEDIUM' | 'OPPORTUNITY', 'title': ..., 'description': ..., 'kwh_impact': ..., 'action': ...}
        ]
    }
    """
    maintenance_recs = []
    energy_recs = []
    
    data = scored_df if scored_df is not None else df
    
    # ==========================================
    # A. MAINTENANCE RECOMMENDATIONS
    # ==========================================
    
    # 1. High Predicted Fault Risk Lamps
    if 'Risk_Level' in data.columns:
        high_risk = data[data['Risk_Level'] == 'HIGH']
        if not high_risk.empty:
            lamp_ids_sample = ", ".join(high_risk['Lamp_ID'].head(4).tolist())
            maintenance_recs.append({
                'priority': 'URGENT',
                'title': f'Critical Fault Risk Detected on {len(high_risk)} Fixtures',
                'description': f'{len(high_risk)} smart street lights exhibit >= 70% failure probability based on electrical telemetry and operational wear (e.g., {lamp_ids_sample}).',
                'affected_count': len(high_risk),
                'action': 'Schedule immediate physical and electrical inspection to prevent total blackout on targeted segments.'
            })
            
    # 2. Voltage Anomalies / Electrical Instability
    if 'Voltage' in data.columns:
        voltage_faults = data[(data['Voltage'] < 195) | (data['Voltage'] > 245)]
        if not voltage_faults.empty:
            zones_affected = ", ".join(voltage_faults['Zone'].unique()[:3])
            maintenance_recs.append({
                'priority': 'HIGH',
                'title': f'Severe Voltage Sags / Surges in {len(voltage_faults)} Luminaires',
                'description': f'Substations or local feeder lines in {zones_affected} are experiencing abnormal voltage variations (<195V or >245V), degrading driver power electronics.',
                'affected_count': len(voltage_faults),
                'action': 'Deploy line electricians to calibrate feeder taps and test capacitor banks in affected circuits.'
            })
            
    # 3. Repeated Past Faults (Chronic Failures)
    if 'Previous_Faults' in data.columns:
        repeat_faults = data[data['Previous_Faults'] >= 3]
        if not repeat_faults.empty:
            maintenance_recs.append({
                'priority': 'HIGH',
                'title': f'Chronic Repeat Failures on {len(repeat_faults)} Fixtures',
                'description': f'These street lights have logged 3 or more previous maintenance events. Continued patching is cost-ineffective compared to complete assembly replacement.',
                'affected_count': len(repeat_faults),
                'action': 'Replace full luminaire and junction wiring rather than component-level repairs.'
            })
            
    # 4. Near End-of-Life Operating Hours
    if 'Operating_Hours' in data.columns:
        end_of_life = data[data['Operating_Hours'] > 12000]
        if not end_of_life.empty:
            maintenance_recs.append({
                'priority': 'MEDIUM',
                'title': f'Operating Hours Exceeding 12,000 hrs on {len(end_of_life)} Fixtures',
                'description': f'{len(end_of_life)} lamps have exceeded 12,000 cumulative burn hours, approaching rated lumen depreciation and ballast burnout threshold.',
                'affected_count': len(end_of_life),
                'action': 'Add to upcoming quarterly preventive maintenance batch for scheduled bulb/driver refresh.'
            })
            
    # 5. Inactive / Unresponsive Luminaires
    if 'Lamp_Status' in data.columns:
        inactive = data[data['Lamp_Status'].str.lower() == 'inactive']
        if not inactive.empty:
            maintenance_recs.append({
                'priority': 'MEDIUM',
                'title': f'{len(inactive)} Unresponsive or Inactive Luminaires Logged',
                'description': f'{len(inactive)} fixtures report zero draw or disconnected telemetry during scheduled illumination cycles.',
                'affected_count': len(inactive),
                'action': 'Inspect line breaker status and optical photocell sensors for debris or physical disconnection.'
            })

    # ==========================================
    # B. ENERGY OPTIMIZATION RECOMMENDATIONS
    # ==========================================
    
    # 1. Legacy HPS / Metal Halide Retrofit
    if 'Lamp_Type' in data.columns:
        legacy_lamps = data[data['Lamp_Type'].isin(['HPS (High Pressure Sodium)', 'Metal Halide'])]
        if not legacy_lamps.empty:
            avg_legacy = legacy_lamps['Energy_Consumption'].mean() if 'Energy_Consumption' in legacy_lamps.columns else 90.0
            avg_led = data[data['Lamp_Type'].str.contains('LED', case=False, na=False)]['Energy_Consumption'].mean() if 'Energy_Consumption' in data.columns else 28.0
            kwh_diff = max(0.0, (avg_legacy - avg_led) * len(legacy_lamps))
            
            energy_recs.append({
                'priority': 'HIGH',
                'title': f'Phase Out {len(legacy_lamps)} Legacy HPS & Metal Halide Fixtures',
                'description': f'Legacy fixtures draw ~{avg_legacy:.1f} kWh vs ~{avg_led:.1f} kWh for LED luminaires. Modern smart LED retrofits reduce power consumption by ~60-70%.',
                'kwh_impact': f'~{kwh_diff:,.0f} kWh estimated potential monthly savings',
                'action': 'Prioritize high-pressure sodium fixtures along main traffic corridors for municipal LED conversion subsidies.'
            })

    # 2. High Consumption Zones Optimization
    if energy_analysis and energy_analysis.get('high_consumption_zones'):
        high_zones = energy_analysis['high_consumption_zones']
        energy_recs.append({
            'priority': 'HIGH',
            'title': f'Implement Dynamic Scheduling in Top Consumption Zones: {", ".join(high_zones)}',
            'description': f'Statistical analysis reveals {", ".join(high_zones)} exceed the 70th percentile of municipal power consumption due to continuous full-brightness burning.',
            'kwh_impact': f'~{energy_analysis.get("dimming_savings", 1200):,.0f} kWh estimated potential savings',
            'action': 'Deploy IoT dynamic dimming profiles (dim to 40% between 01:00 AM and 05:00 AM when pedestrian and vehicular traffic decreases).'
        })

    # 3. Excessive Operating Hours / Photocell Drift
    if 'Operating_Hours' in data.columns:
        excessive = data[data['Operating_Hours'] > data['Operating_Hours'].quantile(0.85)]
        if not excessive.empty:
            energy_recs.append({
                'priority': 'MEDIUM',
                'title': f'Daylight Sensor Re-Calibration on {len(excessive)} Over-Burning Fixtures',
                'description': f'{len(excessive)} fixtures exhibit abnormal cumulative burn times relative to peers, indicative of faulty photocells failing to switch off at dawn.',
                'kwh_impact': 'Eliminates redundant daytime operational energy wastage',
                'action': 'Clean optical sensor lenses and reset astronomical clock schedules on central controller.'
            })

    # 4. Solar-Hybrid Expansion in Suburban Sectors
    if 'Zone' in data.columns:
        suburban_count = len(data[data['Zone'].str.contains('Suburban|Residential', case=False, na=False)])
        energy_recs.append({
            'priority': 'OPPORTUNITY',
            'title': f'Solar-Hybrid LED Expansion Across {suburban_count} Suburban Poles',
            'description': 'Suburban arterial corridors possess unshaded solar exposure ideal for decentralized solar battery integration with grid fall-back.',
            'kwh_impact': 'Up to 50% grid off-load during summer and peak tariff months',
            'action': 'Conduct pole structural integrity survey for solar panel and lithium battery mounting.'
        })

    return {
        'maintenance_recs': maintenance_recs,
        'energy_recs': energy_recs
    }
