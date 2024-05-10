import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns 
import numpy as np

excel_path_results = Path(f'results/synthesized experiment/synthesized_results.xlsx')

# Set general font size for readability
plt.rcParams.update({
    'font.size': 12,     # Set default font size
    'font.weight': 'bold',  # Set default font weight to bold
    'axes.labelweight': 'bold',  # Ensure the axis labels are bold
    'axes.titleweight': 'bold',  # Ensure the titles are bold
    'figure.titleweight': 'bold',  # Bold for suptitle if you use fig.suptitle()
    'xtick.labelsize': 12,  # Font size for X-tick labels
    'ytick.labelsize': 12,  # Font size for Y-tick labels
    'xtick.major.size': 5,  # Length of major ticks
    'ytick.major.size': 5,  # Length of major ticks
    'xtick.minor.size': 3,  # Length of minor ticks
    'ytick.minor.size': 3   # Length of minor ticks
})

xls = pd.ExcelFile(excel_path_results)
sheet_names = xls.sheet_names

num_sheets = len(xls.sheet_names)
fig, axes = plt.subplots(ncols=num_sheets, figsize=(12* num_sheets, 7))  # Adjust figure size as necessary

# Iterate through each sheet in the Excel file
for ax, sheet_name in zip(axes, xls.sheet_names):
    # Read sheet into DataFrame
    df = xls.parse(sheet_name)
    
    # Assuming each row is a method and the rest are results
    # Transposing the DataFrame so that methods become columns
    df_transposed = df.set_index(df.columns[0]).T
    mean_value = np.mean(df.values[:,1:].astype(float))
    # Plotting boxplot using seaborn on a subplot axis
    sns.boxplot(width=.8, data=df_transposed, ax=ax).set(xlabel=' ')

    ax.set_title(f'Synthesized dataset {sheet_names.index(sheet_name) + 1}')
    #ax.set_ylabel('Results')
    ax.tick_params(axis='x', rotation=30)  # Rotates the method names for better visibility

fig.subplots_adjust(top=0.95, bottom=0.17, wspace=0.05, hspace=0.05)

fig.savefig("results/synthesized experiment/synthesized_plot.png", dpi=500, format='png', bbox_inches='tight')

print("done!")
