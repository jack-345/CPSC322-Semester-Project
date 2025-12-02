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

