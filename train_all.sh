#!/bin/bash

# List of Python scripts with their respective arguments

scripts=(
    "train_gmlm.py -c no -s 4 -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 100 -l 5e-5 -b 16"
    "train_gmlm.py -c random -s 4 -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 100 -l 5e-5 -b 16"
    "train_gmlm.py -c ts_blank -s 4 -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 100 -l 5e-5 -b 16"
    "train_gmlm.py -c ts_incr -s 4 -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 100 -l 5e-5 -b 16"
)

# Name of the conda environment
conda_env="torch"

# Loop through the scripts and create a screen for each
for script in "${scripts[@]}"; do
    # Extract the base name of the script (first word) to use as the screen name
    screen_name=$(basename "$(echo $script | awk '{print $1}')" .py)
    
    # Start a new detached screen and execute commands
    screen -dmS "$screen_name" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh;  # Update this path if your conda is located elsewhere
        conda activate $conda_env;
        python $script;
        exec bash
    "
    echo "Started screen '$screen_name' for script '$script'."
done
