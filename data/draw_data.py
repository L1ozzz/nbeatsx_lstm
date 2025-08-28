import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'imputed_data_KNN_1.csv'
data = pd.read_csv(file_path)

# Convert the 'Day(Local_Date)' column to datetime for better plotting
data['Day(Local_Date)'] = pd.to_datetime(data['Day(Local_Date)'])

# Set the 'Day(Local_Date)' as the index for time-series plotting
data.set_index('Day(Local_Date)', inplace=True)

# Create a plot for each column except 'Season'
columns_to_plot = [col for col in data.columns if col != 'Season']

# Adjust layout to multiple columns
num_cols = 3
num_rows = (len(columns_to_plot) + num_cols - 1) // num_cols

plt.figure(figsize=(20, num_rows * 5))

for i, column in enumerate(columns_to_plot, start=1):
    ax = plt.subplot(num_rows, num_cols, i)

    # 1) Plot the line graph (all data)
    ax.plot(data.index, data[column], label=column, linewidth=1.5, color='blue')

    # 2) Calculate the upper and lower bounds (IQR method)
    col_series = data[column].dropna()
    if len(col_series) > 0:
        q1, q3 = col_series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 3) Find outlier positions
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
        outlier_count = outlier_mask.sum()
    else:
        outlier_count = 0

    # 4) Set axis labels with larger font size
    ax.set_xlabel('Date', fontsize=18)  # Increased fontsize for x-axis label
    ax.set_ylabel(column, fontsize=18)  # Increased fontsize for y-axis label

    # Removed legend and title
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5) Add subfigure label (e.g., (a), (b), (c), ...) below x-axis
    label = f"({chr(97 + i)})"  # This will generate (a), (b), (c), ...
    ax.text(0.5, -0.15, label, transform=ax.transAxes, fontsize=18, va='top', ha='center', fontweight='bold')

# Adjust layout to make the plots fit better
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# Save the plot as SVG
plt.savefig('data/dataset.svg', dpi=300, bbox_inches='tight', format='svg')

# Show the plot
plt.show()
