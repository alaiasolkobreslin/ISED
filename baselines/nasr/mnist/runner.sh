#!/bin/bash

# Define a list of random seeds
seeds=(3177 5848 9175 8725 1234 1357 2468 548 6787 8371)

# Define a list of python files
python_files=("sum_2.py" "sum_3.py" "sum_4.py" "not_3_4.py" "mult_2.py" "mod_2.py" "less_than.py" "how_many_3_4.py" "equal.py" "add_sub.py" "add_mod_3.py")

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Loop through each python file
for file in "${python_files[@]}"; do
    # Loop through each seed and execute the command
    for seed in "${seeds[@]}"; do
        output_file="outputs/${file%.py}_seed_${seed}_output.txt"  # Generate output file name
        python -u "$file" --seed "$seed" > "$output_file" 2>&1  # Redirect output to file
    done
done