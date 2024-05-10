import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 

essential = True # sim
ext_name = '_inessential' if not essential else ''

excel_path_results = Path(f'results/game_simulation/results{ext_name}.xlsx')
excel_path_time = Path(f'results/game_simulation/time{ext_name}.xlsx')
save_fig_name = f'results/game_simulation/game_simulation{ext_name}.png'

time_table = False 
plot_performance = True

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


if time_table: 
    xls = pd.ExcelFile(excel_path_time)
    sheet_names = xls.sheet_names

    # Prepare a dictionary to store performance data from each sheet
    performance_data = {}

    # Process each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Get the last column's label, which represents the sample size
        last_column_label = df.columns[-1]
        
        # Extract the last column for performance data
        last_column_data = df[last_column_label]
        
        # Process the data to have "average ± standard deviation" format
        method_data = [f"{last_column_data.iloc[i]:.3f} ± {last_column_data.iloc[i+1]:.3f}" for i in range(0, len(last_column_data), 2)]
        
        # Add the processed data to the dictionary
        performance_data[sheet_name] = method_data

    # Assuming all sheets have the same methods in the same order, get the method names from the first sheet
    methods = pd.read_excel(xls, sheet_name=xls.sheet_names[0]).index[::2].tolist()

    # Begin constructing the LaTeX table
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{l" + "c" * len(performance_data) + "}\n\\hline\nMethod & " + " & ".join(xls.sheet_names) + " \\\\ \\hline\n"

    # Add rows for each method
    for i, method in enumerate(methods):
        row_data = [performance_data[sheet_name][i] for sheet_name in xls.sheet_names]
        latex_table += f"{method} & " + " & ".join(row_data) + " \\\\\n"

    # End the LaTeX table
    latex_table += "\\hline\n\\end{tabular}\n\\caption{Performance of Methods Across Different Sample Sizes}\n\\end{table}"

    # Print the LaTeX table
    print(latex_table)

if plot_performance:
    xls = pd.ExcelFile(excel_path_results)
    sheet_names = xls.sheet_names

    plot_player_no = [11,12,13,14,15] # [7,8,9,10,11,12] # [8,9,10,11,12,13,14,15] # [11,12,13,14,15]

    # Predefined line styles to cycle through
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'purple', 'red']#, , 'yellow', 'black']

    # Determine the subplot grid size (square-like shape preferred)
    n = len(plot_player_no)
    grid_size = 5 #int(n**0.5) + (1 if n**0.5 % 1 else 0)
    rows = int(np.ceil(n / grid_size))
    # Create a figure and subplots
    fig, axs = plt.subplots(rows, grid_size, figsize=(24, 3 * rows))  # Adjust figsize as needed
    axs = axs.flatten()  # Flatten in case of a single row/column of plots
    handles, labels = [], []

    for indx, sheet_name in enumerate(sheet_names):
        df = pd.read_excel(excel_path_results, sheet_name=sheet_name)
        df.set_index(df.columns[0], inplace=True)

        player_no = int(sheet_name.split('_')[1])
        if player_no not in plot_player_no:
            continue

        idx = plot_player_no.index(player_no)

        for method_idx, (i) in enumerate(range(0, df.shape[0], 2)):
            if player_no >= 13 and method_idx == 2: ## skipping the GEM-FIX without regularization for games with more than 13 players
                continue
            averages = df.iloc[i]
            std_devs = df.iloc[i + 1]
            # Choose line style cyclically from the predefined list
            line_style = line_styles[method_idx % len(line_styles)]
            color = colors[method_idx % len(colors)]

            # Plotting on the respective subplot
            lines = axs[idx].errorbar(averages.index, averages, yerr=std_devs, label=df.index[i], fmt=line_style, color=color, capsize=5, elinewidth=2)

            for spine in axs[idx].spines.values():
                spine.set_linewidth(2)
            
            if idx == 0:  # Only add to legend once
                handles.append(lines[0])
                labels.append(df.index[i])
        
        if (idx >= (grid_size)*(rows-1)): 
            axs[idx].set_xlabel('#samples')  # Update this label as needed
        if (idx % grid_size) == 0: axs[idx].set_ylabel('Error')  # Update this label as needed
        axs[idx].set_title(f"#player:{player_no}")
        axs[idx].tick_params(axis='x', rotation=25)

    fig.subplots_adjust(top=0.8, bottom=0.14, wspace=0.15, hspace=0.18)

    # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space at the top for the legend

    fig.legend(handles, labels, loc='upper center', ncol=5)
    plt.show()
    #fig.savefig(save_fig_name, dpi=500, format='png', bbox_inches='tight')

    print("Done")