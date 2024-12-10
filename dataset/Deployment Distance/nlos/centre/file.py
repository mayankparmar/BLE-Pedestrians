import csv

# Function to combine CSV files
def combine_csv_files(input_files, output_file):
    with open(input_files[0], 'r', newline='') as first_file:
        # Read the header from the first file
        header = next(csv.reader(first_file))
    
    combined_rows = [header[:-1]]
    
    for input_file in input_files:
        with open(input_file, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header in the other files
            for row in reader:
                combined_rows.append(row[:-1])  # Exclude the last column
    
    with open(output_file, 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerows(combined_rows)

# List of input CSV files and output CSV file
input_files = ['co_9m_run1.csv', 'co_9m_run2.csv', 'co_9m_run3.csv']
output_file = 'center_o_9m.csv'

# Combine CSV files
combine_csv_files(input_files, output_file)

print(f"Combined files {', '.join(input_files)} into {output_file}")
