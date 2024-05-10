
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the Excel file
file_path = Path('results/tabular experiment/tabular_exp.xlsx')  # Update this to the path of your Excel file
xls = pd.ExcelFile(file_path)
#xls.book.remove(xls.book['Statlog Heart'])
plot_no = len(xls.sheet_names)

# Set general font size for readability
plt.rcParams.update({
    'font.size': 9,     # Set default font size
    'font.weight': 'bold',  # Set default font weight to bold
    'axes.labelweight': 'bold',  # Ensure the axis labels are bold
    'axes.titleweight': 'bold',  # Ensure the titles are bold
    'figure.titleweight': 'bold',  # Bold for suptitle if you use fig.suptitle()
    'xtick.labelsize': 9,  # Font size for X-tick labels
    'ytick.labelsize': 9,  # Font size for Y-tick labels
    'xtick.major.size': 5,  # Length of major ticks
    'ytick.major.size': 5,  # Length of major ticks
    'xtick.minor.size': 3,  # Length of minor ticks
    'ytick.minor.size': 3   # Length of minor ticks
})

# Plotting Performance (Average and Standard Deviation)
fig1, axes1 = plt.subplots(1, plot_no, figsize=(24, 4))  # Single row, six columns for performance plots

# Plotting Execution Time
fig2, axes2 = plt.subplots(1, plot_no, figsize=(24, 4))  # Single row, six columns for execution time plots

line_styles = ['-', '--', '-.', ':', '-.']
colors = ['blue', 'green', 'purple', 'red', 'black']
markers = ['o', 's', '*', '^', 'D', 'v', '>', '<', 'p', 'H']


# Iterate over each dataset (sheet in the Excel file)
handles, handles2 = [], []
labels, labels2 = [], []

for index, sheet_name in enumerate(xls.sheet_names):
    #ax = fig1.add_subplot(2, 3, index + 1)  # 2 rows, 3 columns subplot


    df = pd.read_excel(xls, sheet_name=sheet_name)
    num_samples = df.columns[1:].astype(int)  # Assuming the first column is 'Number of Samples'

    # Iterate over each method, assuming each method occupies three rows
    ax1 = axes1[index]
    for method_index in range(5):  # Adjust the range if there are more or fewer methods
        # if method_index == 2:
        #     continue
        row = method_index * 3
        avg = df.iloc[row, 1:].values     # Average values row
        std = df.iloc[row + 1, 1:].values # Standard deviation row
        exec_time = df.iloc[row + 2, 1:].values # Execution time row

        # Choose line style cyclically from the predefined list
        line_style = line_styles[method_index % len(line_styles)]
        color = colors[method_index % len(colors)]
    
        # Plotting Average Performance and Standard Deviation
        line, caplines, barlinecols = ax1.errorbar(num_samples, avg, yerr=std, label=f"{ df.iloc[:,0].values[row] }", fmt=line_style, color=color, capsize=5, elinewidth=2)
        if index == 0:  # Only capture legend handles and labels once
            handles.append(line)
            labels.append(f'{ df.iloc[:,0].values[row] }')
    
    ax1.set_title(sheet_name)
    ax1.set_xlabel('Samples')
    
    if index == 0:
        ax1.set_ylabel('Absolute error')
        #ax1.legend()

    # Plotting Execution Time on the same graph
    ax2 = axes2[index]
    for method_index in range(5):  # Adjust if different number of methods per dataset
        row = method_index * 3 + 2
        exec_time = df.iloc[row, 1:]

        color = colors[method_index % len(colors)]
        marker = markers[method_index % len(markers)]
        line, = ax2.plot(num_samples, exec_time, label=f"{ df.iloc[:,0].values[row] }", color=color, marker=marker)
        
        if index == 0:  # Only capture legend handles and labels once
            handles2.append(line)
            labels2.append(f'{ df.iloc[:,0].values[row] }')



    ax2.set_title(sheet_name)
    ax2.set_xlabel('Samples')
    
    if index == 0:
        ax2.set_ylabel('Execution Time (s)')


fig1.legend(handles, labels, loc='upper center', ncol=8, frameon=False)
fig2.legend(handles2, labels2, loc='upper center', ncol=8, frameon=False)

fig1.savefig("results/tabular experiment/tabular_sv diff.png", dpi=500, format='png', bbox_inches='tight')
fig2.savefig("results/tabular experiment/tabular_sv time.png", dpi=500, format='png', bbox_inches='tight')


plt.tight_layout()
plt.show()