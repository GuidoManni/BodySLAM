import os

import pandas as pd


def read_and_compile_metrics(input_path, output_path):
    """
    Reads multiple CSV files containing metrics, transposes them,
    and compiles into a single table with environments as rows and metrics as columns.

    Args:
    files_with_envs (list of tuples): A list where each tuple contains the file path and the corresponding environment name.

    Returns:
    pd.DataFrame: A DataFrame with the environment names as rows and metrics as columns.
    """

    paths = sorted(os.listdir(input_path))

    compiled_data = []

    files_with_envs = []

    for folder in paths:
        path_to_csv = os.path.join(input_path, folder) + "/avg.csv"
        env_name = folder
        files_with_envs.append((path_to_csv, env_name))

    for file_path, environment in files_with_envs:
        print(environment)
        # Read the CSV file
        metrics_data = pd.read_csv(file_path, header=None)

        # Transpose the data and set the first row as the header
        metrics_data = metrics_data.transpose()
        metrics_data.columns = metrics_data.iloc[0]
        metrics_data = metrics_data.drop(metrics_data.index[0])

        # Add the environment name
        metrics_data['name_of_the_env'] = environment

        # Append to the list
        compiled_data.append(metrics_data)

    # Concatenate all dataframes
    final_table = pd.concat(compiled_data, ignore_index=True)

    # Save to Excel
    final_table.to_excel(output_path, index=False)

    return final_table

# BodySLAM - Hamlyn
#input_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/Hamlyn_results/"
#output_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM_Hamlyn.xlsx"

# BodySLAM - SCARED
#input_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/SCARED_results/"
#output_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM_SCARED.xlsx"

# EndoDepth - SCARED
#input_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoDepth/results_scared_endo_depth/"
#output_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoDepth_SCARED.xlsx"

# EndoSfmLearner - Hamlyn
#input_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/Hamlyn_results/"
#output_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner_Hamlyn.xlsx"

# EndoSfmLearner - SCARED
input_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/SCARED_results/"
output_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner_SCARED.xlsx"


read_and_compile_metrics(input_path, output_path)