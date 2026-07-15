import csv

def convert_data_to_csv(input_file, output_file, delimiter=','):
    with open(input_file, 'r') as data_file, open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        for line in data_file:
            # Split the line into fields
            fields = line.strip().split(delimiter)
            
            # Write the fields to the CSV file
            csv_writer.writerow(fields)

# Example usage
input_file = 'processed.va.data'
output_file = 'va.csv'
convert_data_to_csv(input_file, output_file)

print(f"Conversion complete. CSV file saved as {output_file}")
