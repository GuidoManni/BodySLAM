#!/bin/bash

# Define your variables
training_dataset_path="/path/to/training/dataset"
testing_dataset_path="/path/to/testing/dataset"
num_epoch="100"
batch_size="4"
path_to_the_model="/path/to/model"
load_model="True"
standard_cycle="value"
standard_identity="value"
weight_id_loss="value"
weight_cycle_loss="value"
id="unique_id"
id_wandb_run="wandb_run_id"

# Activate the virtual environment
source /home/gvide/PycharmProjects/SurgicalSlam/venv/bin/activate

# Run the Python script with provided arguments
python MPEM/train_script.py \
    --training_dataset_path $training_dataset_path \
    --testing_dataset_path $testing_dataset_path \
    --num_epoch $num_epoch \
    --batch_size $batch_size \
    --path_to_the_model $path_to_the_model \
    --load_model $load_model \
    --standard_cycle $standard_cycle \
    --standard_identity $standard_identity \
    --weights_id_loss $weight_id_loss \
    --weights_cycle_loss $weight_cycle_loss \
    --id $id \
    --id_wandb_run $id_wandb_run

# Deactivate the virtual environment (optional)
deactivate
