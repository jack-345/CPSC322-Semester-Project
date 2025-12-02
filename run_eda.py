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

        
                       




