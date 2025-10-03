import pandas as pd
import numpy as np

input_file = 'powergrid_material_demand_latest.csv'
output_file = 'powergrid_material_demand_latest.csv'
df = pd.read_csv(input_file)

def calc_conductor_extra(row):
    tower = row['Tower_Type'].strip().upper()
    
    if 'S/C' in tower:
        base = row['Line_Length_CKM'] * 3
    elif 'D/C' in tower:
        base = row['Line_Length_CKM'] * 6
    else:
        base = row['Line_Length_CKM'] * 3  
    
    noise = np.random.uniform(0, 0.05 * base)
    return int(base + noise)

df['Conductor_Demand_Km'] = df.apply(calc_conductor_extra, axis=1)

df.to_csv(output_file, index=False)
print(f"Updated dataset saved as '{output_file}' with corrected Conductor_Demand_Km (always ≥ minimum).")
