#!/bin/bash

# List of Python scripts with their respective arguments

scripts=(
    "train_gmlm.py -c random -f CA -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 50 -l 1e-4 -b 20"
    "train_gmlm.py -c base2 -f CA -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 0 -e 50 -l 1e-4 -b 20"
    "train_gmlm_no_pcs.py -c base2 -f CA -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 50 -l 1e-4 -b 20"
    "train_gmlm_no_stage.py -c base2 -f CA -d /media/maindisk/maximos/data/hooktheory_train -v /media/maindisk/maximos/data/hooktheory_test -g 1 -e 50 -l 1e-4 -b 20"
)

# scripts=(
#     # "train_gmlm.py -c random -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 0 -e 15 -l 1e-4 -b 32"
#     # "train_gmlm.py -c base2 -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 1 -e 15 -l 1e-4 -b 32"
#     "train_gmlm_no_stage.py -c base2 -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 0 -e 15 -l 1e-4 -b 40"
#     "train_gmlm_no_pcs.py -c base2 -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 1 -e 15 -l 1e-4 -b 40"
#     "train_gmlm_no_stage.py -c random -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 2 -e 15 -l 1e-4 -b 20"
#     "train_gmlm_no_pcs.py -c random -d /media/maindisk/maximos/data/hooktheory_all12_train -v /media/maindisk/maximos/data/hooktheory_all12_test -g 2 -e 15 -l 1e-4 -b 20"
# )

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
