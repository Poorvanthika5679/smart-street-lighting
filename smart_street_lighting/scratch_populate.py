import random
from datetime import datetime, timedelta

zones = [
    'Downtown Commercial', 'North Residential', 'Industrial Corridor',
    'Tech Park Sector 5', 'Harbor Waterfront', 'Suburban Ring'
]
lamp_types = [
    ('LED (Smart)', 28.0, 0.38, 0.08),
    ('HPS (High Pressure Sodium)', 82.0, 1.15, 0.24),
    ('Metal Halide', 90.0, 1.25, 0.28),
    ('Solar-Hybrid LED', 14.0, 0.22, 0.05)
]

random.seed(42)
rows = ["Lamp_ID,Zone,Lamp_Type,Lamp_Status,Operating_Hours,Energy_Consumption,Date_Time,Voltage,Current,Previous_Faults,Lamp_Age,Fault_Status"]
base_date = datetime(2026, 8, 1, 0, 0, 0)

for i in range(1, 651):
    lid = f"SL-{1000 + i}"
    zone = random.choice(zones)
    # pick lamp type with realistic distribution
    r = random.random()
    if r < 0.48:
        ltype, base_kwh, base_amp, f_bias = lamp_types[0]
    elif r < 0.72:
        ltype, base_kwh, base_amp, f_bias = lamp_types[1]
    elif r < 0.88:
        ltype, base_kwh, base_amp, f_bias = lamp_types[2]
    else:
        ltype, base_kwh, base_amp, f_bias = lamp_types[3]
        
    if "LED" in ltype:
        age = random.randint(3, 36)
        hours = round(random.gauss(4200, 1500), 1)
    else:
        age = random.randint(24, 72)
        hours = round(random.gauss(10500, 2800), 1)
    hours = max(200.0, hours)
    
    if age > 40 or hours > 10000:
        pf = random.choices([0, 1, 2, 3, 4], weights=[0.25, 0.35, 0.22, 0.12, 0.06])[0]
    else:
        pf = random.choices([0, 1, 2], weights=[0.8, 0.16, 0.04])[0]
        
    # Voltage
    if random.random() < 0.08:
        volt = round(random.choice([random.uniform(178.0, 194.0), random.uniform(246.0, 258.0)]), 1)
    else:
        volt = round(random.gauss(224.5, 3.5), 1)
        
    # Energy
    z_mult = 1.25 if zone in ['Downtown Commercial', 'Industrial Corridor'] else (0.85 if zone == 'Tech Park Sector 5' else 1.0)
    kwh = round(max(5.0, random.gauss(base_kwh * z_mult, base_kwh * 0.15)), 2)
    amp = round(max(0.1, random.gauss(base_amp, 0.05)), 2)
    if volt < 195.0:
        amp = round(amp * 1.35, 2)
        
    dt = (base_date + timedelta(days=random.randint(0, 45), hours=random.randint(18, 23), minutes=random.choice([0, 15, 30, 45]))).strftime("%Y-%m-%d %H:%M:%S")
    
    # Fault probability
    risk = f_bias * 0.35 + (0.35 if (volt < 195 or volt > 245) else 0) + (pf / 4.0) * 0.2 + min(1.0, hours / 15000.0) * 0.15 + (0.15 if zone == 'Harbor Waterfront' else 0)
    f_stat = 1 if random.random() < min(0.92, max(0.03, risk)) else 0
    
    if f_stat == 1:
        stat = random.choices(['Active', 'Maintenance Required', 'Inactive'], weights=[0.35, 0.55, 0.10])[0]
    else:
        stat = random.choices(['Active', 'Inactive'], weights=[0.95, 0.05])[0]
        
    rows.append(f"{lid},{zone},{ltype},{stat},{hours},{kwh},{dt},{volt},{amp},{pf},{age},{f_stat}")

with open("data/sample_street_lights.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")
print(f"Done: {len(rows)-1} rows written.")
