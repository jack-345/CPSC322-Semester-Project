import csv 
import matplotlib.pyplot as plt


def load_data(filename):
    table = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            table.append(row)
    return header, table

def get_column_numeric(table,header,col_name):
    col_index = header.index(col_name)
    values = []
    for row in table:
        try:
            values.append(float(row[col_index]))
        except ValueError:
            pass
    return values

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    return [row[col_index] for row in table]


def get_frequency_table(values):
    freq_dict = {}
    for value in values:
        freq_dict[value] = freq_dict.get(value,0) + 1
    return freq_dict 

def compute_percentile(values,percentile):
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile/100
    lower = int(index)
    upper = lower + 1
    weight = index - lower 
    if upper >= len(sorted_values):
        return sorted_values[lower]
    return sorted_values[int(lower)]* (1- weight) + sorted_values[int(upper)] * weight

filename = 'climate_change_impact_on_agriculture_2024.csv'
header,table = load_data(filename)

print("=" * 70)
print("Exploratory Data Analysis: Climate Change Impact on Agriculture")
print("Team: Kim Lenz and Jack Oh")
print("=" * 70)
print(f"Dataset: {len(table)} instances, {len(header)} attributes")

#target variable for crop yield value into categories : Assign High, Medium, or Low Crop yield --> dealing with the continuous variable issue

crop_yield = get_column_numeric(table, header, 'Crop_Yield_MT_per_HA')
percent_33 = compute_percentile(crop_yield,33)
percent_67 = compute_percentile(crop_yield,67)

print(f"\n Target Variable: Yield Category")
print(f"Low: {percent_33:.2f} MT/HA")
print(f"Medium: {percent_33:.2f} - {percent_67:.2f} MT/HA")
print(f"High: {percent_67:.2f} MT/HA")


yield_catgory = []
for row in table:
    try:
        yield_value = float(row[header.index('Crop_Yield_MT_per_HA')])
        if yield_value < percent_33:
            yield_catgory.append('Low')
        elif yield_value < percent_67:
            yield_catgory.append('Medium')
        else:
            yield_catgory.append('High')
    except:
        yield_catgory.append('Unknown')

category_count = get_frequency_table(yield_catgory)
print("\n Class Distribution: ") 
for category in ['Low', 'Medium', 'High']:
    count = category_count.get(category, 0)
    percentage = (count/len(yield_catgory)) * 100
    print(f" {category}: {count} ({percentage:.1f}%)")


import csv

def load_data(filename):
    table = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            table.append(row)
    return header, table

def compute_summary_stats(values):
    """Calculate summary statistics for a list of numeric values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    range_val = max_val - min_val
    mean = sum(sorted_vals) / n
    
    
    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        median = sorted_vals[n//2]
    
    
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1 = sorted_vals[q1_idx]
    q3 = sorted_vals[q3_idx]
    iqr = q3 - q1
    
    
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    std_dev = variance ** 0.5
    
    return {
        'count': n,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'mean': mean,
        'median': median,
        'std': std_dev,
        'Q1': q1,
        'Q2': median,
        'Q3': q3,
        'IQR': iqr
    }


filename = 'climate_change_impact_on_agriculture_2024.csv'
header, table = load_data(filename)


numeric_features = [
    'Average_Temperature_C', 'Total_Precipitation_mm', 
    'CO2_Emissions_MT', 'Crop_Yield_MT_per_HA',
    'Extreme_Weather_Events', 'Irrigation_Access_%', 
    'Pesticide_Use_KG_per_HA', 'Fertilizer_Use_KG_per_HA', 
    'Soil_Health_Index'
]

print("=" * 70)
print("SUMMARY STATISTICS FOR NUMERIC FEATURES")
print("=" * 70)

for feat_name in numeric_features:
    feat_idx = header.index(feat_name)
    values = []
    for row in table:
        try:
            values.append(float(row[feat_idx]))
        except:
            pass
    
    stats = compute_summary_stats(values)
    
    print(f"\n{feat_name}:")
    print(f"  Count:  {stats['count']}")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std:    {stats['std']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Q1:     {stats['Q1']:.2f}")
    print(f"  Q2:     {stats['Q2']:.2f}")
    print(f"  Q3:     {stats['Q3']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  Range:  {stats['range']:.2f}")
    print(f"  IQR:    {stats['IQR']:.2f}")

    #summary stats created with the help of Claude AI and course materials 

# Figure 1
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(crop_yield, bins=30)
plt.axvline(percent_33, color='red', linestyle='--')
plt.axvline(percent_67, color='orange', linestyle='--')
plt.xlabel('Crop Yield')
plt.ylabel('Frequency')
plt.title('Yield Distribution')

plt.subplot(1, 2, 2)
plt.bar(['Low', 'Medium', 'High'], 
        [category_count.get('Low', 0), category_count.get('Medium', 0), category_count.get('High', 0)])
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Class Distribution')

plt.tight_layout()
plt.savefig('figures/figure1_target.pdf')
plt.savefig('figures/figure1_target.png', dpi=150)
print("Figure 1")
plt.close()

# Figure 2: Climate
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(get_column_numeric(table, header, 'Average_Temperature_C'), bins=25)
plt.xlabel('Temperature (°C)')
plt.title('Temperature')

plt.subplot(2, 2, 2)
plt.hist(get_column_numeric(table, header, 'Total_Precipitation_mm'), bins=25)
plt.xlabel('Precipitation (mm)')
plt.title('Precipitation')

plt.subplot(2, 2, 3)
plt.hist(get_column_numeric(table, header, 'CO2_Emissions_MT'), bins=25)
plt.xlabel('CO2 (MT)')
plt.title('CO2')

plt.subplot(2, 2, 4)
plt.hist(get_column_numeric(table, header, 'Extreme_Weather_Events'), bins=20)
plt.xlabel('Events')
plt.title('Weather Events')

plt.tight_layout()
plt.savefig('figures/figure2_climate.pdf')
plt.savefig('figures/figure2_climate.png', dpi=150)
print(" Figure 2")
plt.close()

# Figure 3: Agriculture
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(get_column_numeric(table, header, 'Irrigation_Access_%'), bins=25)
plt.xlabel('Irrigation (%)')
plt.title('Irrigation')

plt.subplot(2, 2, 2)
plt.hist(get_column_numeric(table, header, 'Pesticide_Use_KG_per_HA'), bins=25)
plt.xlabel('Pesticide')
plt.title('Pesticide')

plt.subplot(2, 2, 3)
plt.hist(get_column_numeric(table, header, 'Fertilizer_Use_KG_per_HA'), bins=25)
plt.xlabel('Fertilizer')
plt.title('Fertilizer')

plt.subplot(2, 2, 4)
plt.hist(get_column_numeric(table, header, 'Soil_Health_Index'), bins=25)
plt.xlabel('Soil Health')
plt.title('Soil Health')

plt.tight_layout()
plt.savefig('figures/figure3_agriculture.pdf')
plt.savefig('figures/figure3_agriculture.png', dpi=150)
print("[✓] Figure 3")
plt.close()

# Figure 4: Categorical
plt.figure(figsize=(14, 10))
for i, col in enumerate(['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies'], 1):
    plt.subplot(2, 2, i)
    values = get_column(table, header, col)
    freq = get_frequency_table(values)
    top10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    plt.barh([x[0] for x in top10], [x[1] for x in top10])
    plt.xlabel('Count')
    plt.title(f'{col}')

plt.tight_layout()
plt.savefig('figures/figure4_categorical.pdf')
plt.savefig('figures/figure4_categorical.png', dpi=150)
print("Figure 4")
plt.close()

# Figure 5: Box plots
plt.figure(figsize=(14, 10))
for i, (col, label) in enumerate([
    ('Average_Temperature_C', 'Temp'),
    ('Total_Precipitation_mm', 'Precip'),
    ('Soil_Health_Index', 'Soil'),
    ('Irrigation_Access_%', 'Irrigation')
], 1):
    plt.subplot(2, 2, i)
    values = get_column_numeric(table, header, col)
    low = [values[j] for j, cat in enumerate(yield_catgory) if cat == 'Low' and j < len(values)]
    med = [values[j] for j, cat in enumerate(yield_catgory) if cat == 'Medium' and j < len(values)]
    high = [values[j] for j, cat in enumerate(yield_catgory) if cat == 'High' and j < len(values)]
    plt.boxplot([low, med, high], labels=['Low', 'Med', 'High'])
    plt.ylabel(label)
    plt.title(f'{label} by Yield')

plt.tight_layout()
plt.savefig('figures/figure5_boxplots.pdf')
plt.savefig('figures/figure5_boxplots.png', dpi=150)
print("Figure 5")
plt.close()

# Figure 6: Scatter
plt.figure(figsize=(14, 10))
colors = {'Low': 'red', 'Medium': 'yellow', 'High': 'green'}

for i, (col, label) in enumerate([
    ('Average_Temperature_C', 'Temp'),
    ('Total_Precipitation_mm', 'Precip'),
    ('Soil_Health_Index', 'Soil'),
    ('Irrigation_Access_%', 'Irrigation')
], 1):
    plt.subplot(2, 2, i)
    values = get_column_numeric(table, header, col)
    for cat in ['Low', 'Medium', 'High']:
        idx = [j for j, c in enumerate(yield_catgory) if c == cat]
        x = [values[j] for j in idx if j < len(values)]
        y = [crop_yield[j] for j in idx if j < len(crop_yield)]
        plt.scatter(x, y, c=colors[cat], label=cat, s=10, alpha=0.5)
    plt.xlabel(label)
    plt.ylabel('Yield')
    plt.title(f'{label} vs Yield')
    plt.legend()

plt.tight_layout()
plt.savefig('figures/figure6_scatter.pdf')
plt.savefig('figures/figure6_scatter.png', dpi=150)
print("Figure 6")
plt.close()


#AI ACKNOWLEDGEMENT 
#graphs created using Claude AI
#editing and corrections of code done using Claude AI







