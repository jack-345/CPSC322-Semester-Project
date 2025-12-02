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
    upper = index - lower
    weight = index - lower 
    if upper >= len(sorted_values):
        return sorted_values[lower]
    return sorted_values[lower] * (1- weight) + sorted_values[upper] * weight

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
        elif yield_value < 67:
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


#Frequency Table for Target Variable Distribution 

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(crop_yield, bins=30, color='skyblue', edgecolor='black')
plt.axvline(percent_33, color='red', linestyle='--', linewidth=2, label='Low/Medium')
plt.axvline(percent_67, color='orange', linestyle='--', linewidth=2, label='Medium/High')
plt.xlabel('Crop Yield (MT per HA)')
plt.ylabel('Frequency')
plt.title('Crop Yield Distribution (Continuous)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
categories = ['Low', 'Medium', 'High']
counts = [category_count.get(cat, 0) for cat in categories]
colors =  ['#ff6b6b','#ffd93d','#6bcf7f']
plt.bar(categories, counts, color=colors, edgecolor='black')
plt.xlabel('Yield Category')
plt.ylabel('Count')
plt.title('Target Variable: Class Distribution')
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure1_target_distribution.pdf')
print("\n Figure 1: Target variable distribution (frequency diagrams)")
plt.show()

#Climate variables -- Frequency Diagram

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
temp = get_column_numeric(table, header, 'Average_Temperature_C')
plt.hist(temp, bins=25, color='coral', edgecolor='black')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.title('Temperature Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
precip = get_column_numeric(table, header, 'Total_Precipitation_mm')
plt.hist(precip, bins=25, color='skyblue', edgecolor='black')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Precipitation Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
co2 = get_column_numeric(table, header, 'CO2_Emissions_MT')
plt.hist(co2, bins=25, color='lightgray', edgecolor='black')
plt.xlabel('CO2 Emissions (MT)')
plt.ylabel('Frequency')
plt.title('CO2 Emissions Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
weather = get_column_numeric(table, header, 'Extreme_Weather_Events')
plt.hist(weather, bins=20, color='salmon', edgecolor='black')
plt.xlabel('Extreme Weather Events')
plt.ylabel('Frequency')
plt.title('Extreme Weather Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure2_climate_distributions.pdf')
print("Figure 2: Climate variable distributions (frequency diagrams)")
plt.show()

#Agriculture variable graphs (soil health, fertilizer, irrigation access, pesticide use, soil health)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
irrigation = get_column_numeric(table, header, 'Irrigation_Access_%')
plt.hist(irrigation, bins=25, color='lightblue', edgecolor='black')
plt.xlabel('Irrigation Access (%)')
plt.ylabel('Frequency')
plt.title('Irrigation Access Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
pesticide = get_column_numeric(table, header, 'Pesticide_Use_KG_per_HA')
plt.hist(pesticide, bins=25, color='lightgreen', edgecolor='black')
plt.xlabel('Pesticide Use (KG/HA)')
plt.ylabel('Frequency')
plt.title('Pesticide Use Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
fertilizer = get_column_numeric(table, header, 'Fertilizer_Use_KG_per_HA')
plt.hist(fertilizer, bins=25, color='wheat', edgecolor='black')
plt.xlabel('Fertilizer Use (KG/HA)')
plt.ylabel('Frequency')
plt.title('Fertilizer Use Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
soil = get_column_numeric(table, header, 'Soil_Health_Index')
plt.hist(soil, bins=25, color='sandybrown', edgecolor='black')
plt.xlabel('Soil Health Index')
plt.ylabel('Frequency')
plt.title('Soil Health Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure3_agricultural_distributions.pdf')
print("Figure 3: Agricultural variable distributions (frequency diagrams)")
plt.show()


#Categorical Variables (country, region, crop type, adaption strategies)

plt.figure(figsize=(14, 10))

categorical_cols = ['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies']

for i, col_name in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)
    
    values = get_column(table, header, col_name)
    freq_dict = get_frequency_table(values)
    sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    
    plt.barh(labels, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Count')
    plt.ylabel(col_name.replace('_', ' '))
    plt.title(f'Top 10 {col_name.replace("_", " ")}')
    plt.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure4_categorical_distributions.pdf')
print("Figure 4: Categorical variable distributions (frequency diagrams)")
plt.show()

#Figure 5 --> box and whisker plot for yield categories 

plt.figure(figsize=(14, 10))

key_vars = [
    ('Average_Temperature_C', 'Temperature (°C)'),
    ('Total_Precipitation_mm', 'Precipitation (mm)'),
    ('Soil_Health_Index', 'Soil Health Index'),
    ('Irrigation_Access_%', 'Irrigation Access (%)')
]

for i, (col_name, label) in enumerate(key_vars, 1):
    plt.subplot(2, 2, i)
    
    # Get data for each category
    values = get_column_numeric(table, header, col_name)
    low_vals = [values[j] for j, cat in enumerate(yield_category) if cat == 'Low' and j < len(values)]
    med_vals = [values[j] for j, cat in enumerate(yield_category) if cat == 'Medium' and j < len(values)]
    high_vals = [values[j] for j, cat in enumerate(yield_category) if cat == 'High' and j < len(values)]
    
    # Create box plot
    data_to_plot = [low_vals, med_vals, high_vals]
    bp = plt.boxplot(data_to_plot, labels=['Low', 'Medium', 'High'],
                     patch_artist=True)
    
    # Color the boxes
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Yield Category')
    plt.ylabel(label)
    plt.title(f'{label} by Yield Category')
    plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure5_box_whisker_plots.pdf')
print("Figure 5: Box and whisker plots (comparison by yield category)")
plt.show()


# Scatter Plots and potentially interesting relationships 
plt.figure(figsize=(14, 10))

color_map = {'Low': 'red', 'Medium': 'yellow', 'High': 'green'}

# Temperature vs Yield
plt.subplot(2, 2, 1)
temp = get_column_numeric(table, header, 'Average_Temperature_C')
for category in ['Low', 'Medium', 'High']:
    indices = [i for i, cat in enumerate(yield_catgory) if cat == category]
    cat_temps = [temp[i] for i in indices if i < len(temp)]
    cat_yields = [crop_yield[i] for i in indices if i < len(crop_yield)]
    plt.scatter(cat_temps, cat_yields, c=color_map[category], 
                label=category, alpha=0.5, s=20)
plt.xlabel('Temperature (°C)')
plt.ylabel('Crop Yield (MT/HA)')
plt.title('Temperature vs Crop Yield')
plt.legend()
plt.grid(True, alpha=0.3)

# Precipitation vs Yield
plt.subplot(2, 2, 2)
precip = get_column_numeric(table, header, 'Total_Precipitation_mm')
for category in ['Low', 'Medium', 'High']:
    indices = [i for i, cat in enumerate(yield_catgory) if cat == category]
    cat_precips = [precip[i] for i in indices if i < len(precip)]
    cat_yields = [crop_yield[i] for i in indices if i < len(crop_yield)]
    plt.scatter(cat_precips, cat_yields, c=color_map[category], 
                label=category, alpha=0.5, s=20)
plt.xlabel('Precipitation (mm)')
plt.ylabel('Crop Yield (MT/HA)')
plt.title('Precipitation vs Crop Yield')
plt.legend()
plt.grid(True, alpha=0.3)

# Soil Health vs Yield
plt.subplot(2, 2, 3)
soil = get_column_numeric(table, header, 'Soil_Health_Index')
for category in ['Low', 'Medium', 'High']:
    indices = [i for i, cat in enumerate(yield_catgory) if cat == category]
    cat_soils = [soil[i] for i in indices if i < len(soil)]
    cat_yields = [crop_yield[i] for i in indices if i < len(crop_yield)]
    plt.scatter(cat_soils, cat_yields, c=color_map[category], 
                label=category, alpha=0.5, s=20)
plt.xlabel('Soil Health Index')
plt.ylabel('Crop Yield (MT/HA)')
plt.title('Soil Health vs Crop Yield')
plt.legend()
plt.grid(True, alpha=0.3)

# Irrigation vs Yield
plt.subplot(2, 2, 4)
irrigation = get_column_numeric(table, header, 'Irrigation_Access_%')
for category in ['Low', 'Medium', 'High']:
    indices = [i for i, cat in enumerate(yield_catgory) if cat == category]
    cat_irrig = [irrigation[i] for i in indices if i < len(irrigation)]
    cat_yields = [crop_yield[i] for i in indices if i < len(crop_yield)]
    plt.scatter(cat_irrig, cat_yields, c=color_map[category], 
                label=category, alpha=0.5, s=20)
plt.xlabel('Irrigation Access (%)')
plt.ylabel('Crop Yield (MT/HA)')
plt.title('Irrigation Access vs Crop Yield')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure6_scatter_relationships.pdf')
print("Figure 6: Scatter plots showing interesting relationships")
plt.show()








