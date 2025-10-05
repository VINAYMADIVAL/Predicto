import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('powergrid_material_demand_latest.csv')

# Mapping: (Tower_Type, Voltage) -> discs per string
disc_map = {
    ('S/C 220 kV', 220): 8,
    ('S/C 400 kV', 400): 12,
    ('S/C 765 kV', 765): 25,
    ('D/C 220 kV', 220): 8,
    ('D/C 400 kV', 400): 12,
    ('D/C 765 kV', 765): 25
}

def calculate_insulators(row):
    # Extract voltage number from Tower_Type string
    voltage = int(row['Tower_Type'].split()[1])
    discs = disc_map.get((row['Tower_Type'], voltage), 8)  # default 8 if not found

    # Phases and circuits
    if 'S/C' in row['Tower_Type']:
        phases = 3
        circuits = 1
    elif 'D/C' in row['Tower_Type']:
        phases = 3
        circuits = 2
    else:
        phases = 3
        circuits = 1

    insulators_per_tower = phases * circuits * discs
    total_insulators = row['Tower_Count'] * insulators_per_tower

    # Add +5% random safety margin
    noise = np.random.normal(0, 0.05 * total_insulators)
    total_insulators += int(abs(noise))  # ensure never less than calculated
    return total_insulators

# Apply function and overwrite the existing column
df['Insulator_Demand_Nos'] = df.apply(calculate_insulators, axis=1)

# Save updated CSV
df.to_csv('powergrid_material_demand_latest.csv', index=False)
print("Updated Insulator_Demand_Nos with safety margin and saved!")
