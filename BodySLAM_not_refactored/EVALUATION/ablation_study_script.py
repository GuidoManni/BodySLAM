import csv
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt



def compute_average_and_save(file_paths, output_file_path):
    # Dictionary to hold the sum of metric values and count for each metric
    metrics_sum = defaultdict(lambda: [0, 0])  # [sum, count]

    # Process each file
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for metric, value in reader:
                # Parse the value string to extract the numbers
                numbers = eval(value)
                metrics_sum[metric][0] += sum(numbers)
                metrics_sum[metric][1] += len(numbers)

    # Compute the average for each metric
    averages = {metric: sum_count[0] / sum_count[1] for metric, sum_count in metrics_sum.items()}

    # Save the averages in a new CSV file
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Average'])
        for metric, average in averages.items():
            writer.writerow([metric, average])


def combine_averages(file_paths, output_file_path):
    combined_averages = defaultdict(list)

    # Process each file
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for metric, average in reader:
                combined_averages[metric].append(average)

    # Save the combined averages in a new CSV file
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header with file names
        writer.writerow(['Metric'] + [file_paths[i].split("/")[-2].replace("_Results", "") for i in range(len(file_paths))])
        for metric, averages in combined_averages.items():
            writer.writerow([metric] + averages)


phase_1 = False
if phase_1:
    # ENDOSLAM
    path_to_file1 = "/home/gvide/Scrivania/Models/Model_16_Results/Results_EndoSLAM"
    output_path_file1 = "/home/gvide/Scrivania/Models/Model_16_Results/EndoSLAM.csv"

    # SCARED
    path_to_file2 = "/home/gvide/Scrivania/Models/Model_16_Results/Results_SCARED"
    output_path_file2 = "/home/gvide/Scrivania/Models/Model_16_Results/SCARED.csv"

    content1 = os.listdir(path_to_file1)

    full_path1 = []
    for file in content1:
        full_path1.append(os.path.join(path_to_file1, file))

    content2 = os.listdir(path_to_file2)

    full_path2 = []
    for file in content2:
        full_path2.append(os.path.join(path_to_file2, file))

    compute_average_and_save(full_path1, output_path_file1)
    compute_average_and_save(full_path2, output_path_file2)

phase_2 = False

if phase_2:
    path_to_file_E = "/home/gvide/Scrivania/Models/Model_%_Results/EndoSLAM.csv"
    path_to_file_S = "/home/gvide/Scrivania/Models/Model_%_Results/SCARED.csv"
    file_paths_E = []
    file_paths_S = []
    # build the path
    for i in range(16):
        i += 1
        file_paths_E.append(path_to_file_E.replace("%", str(i)))
        file_paths_S.append(path_to_file_S.replace("%", str(i)))

    output_path_file_E = "/home/gvide/Scrivania/Models/EndoSLAM_avg_across_models.csv"
    output_path_file_S = "/home/gvide/Scrivania/Models/SCARED_avg_across_models.csv"

    combine_averages(file_paths_E, output_path_file_E)
    combine_averages(file_paths_S, output_path_file_S)

phase_3 = True

if phase_3:
    input_file1 = "/home/gvide/Scrivania/Models/EndoSLAM_avg_across_models.csv"
    input_file2 = "/home/gvide/Scrivania/Models/SCARED_avg_across_models.csv"

    # Reading the datasets
    scared_data = pd.read_csv(input_file2)
    endoslam_data = pd.read_csv(input_file1)

    # Computing a combined average for each model
    combined_averages = merged_data.groupby('Model').mean()

    # Adding a new column for the overall average across both datasets
    combined_averages['Overall_Average'] = combined_averages.mean(axis=1)

    # Finding the model with the lowest overall average
    best_overall_model = combined_averages['Overall_Average'].idxmin()

    # Defining different markers for each metric
    markers = ['o', 's', '^', 'D', 'x', '+', '*']

    # Plotting the filtered data separately for each dataset with different markers
    plt.figure(figsize=(15, 10))

    # Plot for SCARED dataset
    plt.subplot(2, 1, 1)
    for i, metric in enumerate(filtered_data['Metric'].unique()):
        subset = filtered_data[filtered_data['Metric'] == metric]
        plt.scatter(subset['Model'], subset['SCARED_Average'], label=metric, marker=markers[i % len(markers)])
    plt.title('SCARED Dataset Performance')
    plt.xticks(rotation=90)
    plt.ylabel('Average Value')
    plt.legend(title='Metric')

    # Plot for EndoSLAM dataset
    plt.subplot(2, 1, 2)
    for i, metric in enumerate(filtered_data['Metric'].unique()):
        subset = filtered_data[filtered_data['Metric'] == metric]
        plt.scatter(subset['Model'], subset['EndoSLAM_Average'], label=metric, marker=markers[i % len(markers)])
    plt.title('EndoSLAM Dataset Performance')
    plt.xticks(rotation=90)
    plt.ylabel('Average Value')
    plt.legend(title='Metric')

    plt.tight_layout()

