import os
import pandas as pd
import sys

# Set the directory containing the .tsv files
input_directory = sys.argv[1]
output_file = sys.argv[2]
collumn_name = "pairwise potential energy (unweighted)"

# Initialize a list to collect the results
results = []

# Loop through each .tsv file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".tsv"):
        # Read the file
        file_path = os.path.join(input_directory, filename)
        data = pd.read_csv(file_path, sep='\t')

        # Sort the data by 'total weighted energy' in ascending order
        sorted_data = data.sort_values(by=collumn_name)

        # Calculate the average of the 'total weighted energy' for the top 5 entries
        top_5_average = sorted_data[collumn_name].head(5).mean()

        # Add the filename (without .tsv) and the average to the results list
        results.append([filename.replace('.tsv', ''), top_5_average])

# Sort results by the average total weighted energy in ascending order
results.sort(key=lambda x: x[1])

# Write the sorted results to the output file
with open(output_file, 'w') as f:
    # Write the header
    f.write("File Name\t"+collumn_name+"\n")
    # Write each sorted result row
    for result in results:
        f.write(f"{result[0]}\t{result[1]:.2f}\n")

print("Averages calculated and saved to", output_file, "in ascending order of energy.")
