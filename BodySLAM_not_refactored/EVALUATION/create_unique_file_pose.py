import os

import pandas as pd

def process_and_save_metrics(file_paths, output_excel_path):
    """
    Processes multiple metric files and saves a compiled table to an Excel file.

    :param file_paths: List of file paths for the metric files.
    :param output_excel_path: Path to save the output Excel file.
    """
    data = []

    file_path = sorted(os.listdir(file_paths))

    for file in file_path:
        # Read the file
        df = pd.read_csv(os.path.join(file_paths, file))

        # Extract the environment name from the file name
        env_name = os.path.basename(file).split('.')[0]

        # Extract the metrics
        metrics = df.set_index('Metric')['Value'].to_dict()

        # Compile the data
        data.append({
            'name_of_the_env': env_name,
            'ATE': metrics.get('ATE', None),
            'RTE': metrics.get('RTE', None),
            'RRE': metrics.get('RRE', None)
        })

    # Create a DataFrame
    result_df = pd.DataFrame(data)

    # Save to Excel
    result_df.to_excel(output_excel_path, index=False)

    return result_df


# MPEM - EndoSLAM
#file_paths = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/BodySLAM/EndoSLAM_Results/"
#output_execel_path = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/BodySLAM.xlsx"

# MPEM - SCARED
file_paths = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/BodySLAM/SCARED_Results/"
output_execel_path = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/BodySLAM_SCARED.xlsx"

# EndoSfmLearner - EndoSLAM
#file_paths = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoSfmLearner/EndoSLAM/Final_Results/"
#output_execel_path = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoSfmLearner.xlsx"

# EndoSfmLearner - SCARED
#file_paths = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoSfmLearner/SCARED/Final_Results/"
#output_execel_path = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoSfmLearner_SCARED.xlsx"

# EndoDepth - EndoSLAM
#file_paths = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoDepth/EndoSLAM/Final_Results"
#output_execel_path = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoDepth_EndoSLAM.xlsx"

# EndoDepth - SCARED
#file_paths = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoDepth/SCARED/Final_Results"
#output_execel_path = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoDepth_SCARED.xlsx"

process_and_save_metrics(file_paths, output_execel_path)