import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import seaborn as sns

# Constants
thresholds = [0.2, 0.3, 0.4, 0.5]#, 0.7, 0.8]
threshold_labels = ['20%', '30%', '40%', '50%']

line_styles = ['-', '--', '-.', ':']
colors = ['blue', 'green', 'purple', 'red', 'black', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o', 's', '*', '^', 'D', 'v', '>', '<', 'p', 'H']

# Set general font size for readability
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
fig1, axes1 = plt.subplots(1, 6, figsize=(20, 4))  # Single row, six columns for performance plots

# Plotting Execution Time
fig2, axes2 = plt.subplots(1, 6, figsize=(20, 4))  # Single row, six columns for execution time plots

line_styles = ['-', '--', '-.', ':']
colors = ['blue', 'green', 'purple', 'red', 'black']
colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728',  
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

markers = ['o', 's', '*', '^', 'D', 'p', 'H', 'v', '>', '<']

handles, labels = [] , []

# Load Excel file
datasets = ['text_exp_amazon polarity', 'text_exp_go_emotion', 'text_exp_imdb', 'text_exp_rotten_tomato', 'text_exp_sst2', 'text_exp_yelp_review']
name = ['Amazon Polarity', 'GoEmotions', 'IMDB', 'Rotten Tomato', 'GLUE SST2', "Yelp Review Full"]
for i, ds_name in enumerate(datasets):
    file_path = Path(f'results/textual experiment/{ds_name}.xlsx')
    xls = pd.ExcelFile(file_path)

    # Initialize results dictionary for all sheets
    all_sheets_stats = {}

    # Process each sheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name, header=None)
        stats = {t_label: [] for t_label in threshold_labels}  # Initialize stats for this sheet

        time = df.iloc[:,-2:-1].values.squeeze()

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            total_features = int(row.iloc[-1])  # Assuming 'Total_Features' is the last column
            valid_columns_count = int(total_features * 0.95)  # 95% of total features have values

            # Compute mean performances for specified thresholds
            for t, label in zip(thresholds, threshold_labels):
                max_index = int(total_features * t)
                valid_data = row[:valid_columns_count].dropna()  # Ensure no NaN values are included
                relevant_data = valid_data.iloc[:max_index]
                stats[label].append(relevant_data.mean())

        # Calculate overall stats for the current sheet
        mean_stats = {label: np.mean(stats[label]) for label in threshold_labels}
        std_stats = {label: np.std(stats[label]) for label in threshold_labels}

        
        all_sheets_stats[sheet_name] = {'means': mean_stats, 'stds': std_stats, 'time': time}

    # Plotting the performance
    ax1 = axes1[i]
    for index, (sheet_name, data) in enumerate(all_sheets_stats.items()):
        means = [data['means'][label] for label in threshold_labels]
        stds = [data['stds'][label] for label in threshold_labels]
        ind = np.arange(len(threshold_labels))  # the    x locations for the groups
        
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        line_style = line_styles[index % len(line_styles)]


        # Plotting
        line = ax1.plot(ind, means,label=sheet_name, markersize=6, color=color, marker=marker)


    # Add some text for labels, title and axes ticks
    #ax1.set_xlabel('Percentage of Removed Features')
    if i == 0: ax1.set_ylabel('Average difference')
    xlabels = ['0']
    xlabels.extend(threshold_labels)
    ax1.set_xticklabels(xlabels)
    ax1.set_title(name[i])

    # Plotting
    ax2 = axes2[i]
    exec_time = []
    methods = []
    for index, (sheet_name, data) in enumerate(all_sheets_stats.items()):
        exec_time.append(data['time'])
        methods.append(sheet_name)
    
    mean_values = [np.mean(item) for item in exec_time]
    error_values = [np.std(item) for item in exec_time]
    box = ax2.boxplot(exec_time, patch_artist=True, notch=False, vert=1, widths=0.4, labels=methods) #methods, mean_values, yerr=error_values, capsize=5, color='skyblue', align='center', alpha=0.7, ecolor='black')
    
    ax2.set_title(name[i])

    import matplotlib.patches as mpatches

    # Coloring each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set custom x-axis labels
    custom_labels = [''] * 7 # ['E 1', 'E 2', 'E 3', 'E 4', 'E 5']
    ax2.set_xticklabels(custom_labels)

    # # Creating a custom legend
    if i == 2:
        legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, methods)]
        legend = ax2.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(1.12, 1.18), ncol=8) 

handles, labels = ax1.get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', ncol=12, frameon=False)
fig1.text(0.5, 0.02, 'Percentage of Removed Feature', ha='center', fontweight='bold')

handles, labels = ax2.get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', ncol=18, frameon=False)


#### print
fig1.savefig("results/textual experiment/textual feature_removal.png", dpi=500, format='png', bbox_inches='tight')
fig2.savefig("results/textual experiment/textual time.png", dpi=500, format='png', bbox_inches='tight')

print("done!")
