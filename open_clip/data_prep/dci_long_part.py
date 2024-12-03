import json
from glob import glob
import pandas as pd
import argparse
import shutil
from tqdm import tqdm
import os

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dci_json_folder', type=str, default='/DCI/data/densely_captioned_images/annotations/')
    parser.add_argument('--output_csv_path', type=str, default='/data/dci_long')
    return parser.parse_args()

# Function to read JSON file
def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Function to write JSON file (not used in the main script)
def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    dci_json_folder = args.dci_json_folder
    output_csv_path = args.output_csv_path

    # make if output_csv_path does not exist
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)

    # Get list of JSON files in the specified folder
    json_files = glob(dci_json_folder + '/*.json')
    # sort json_files
    json_files.sort()

    # Initialize empty list to store processed data
    data = []

    # Process each JSON file with a progress bar
    for json_file in tqdm(json_files, desc='Processing JSON files'):
        json_data = read_json(json_file)
        
        # Extract relevant information
        long_caption = json_data['extra_caption']
        if len(long_caption) == 0:
            continue
        short_caption = json_data['short_caption'] 
        image_path = json_data['image']
        
        # Append processed data to the list
        data.append({
            'filepath': image_path,
            'title': long_caption
        })

    # Create DataFrame from processed data
    df = pd.DataFrame(data)

    # print how many long captions are available
    print(f"Number of long captions: {len(df)}")
    
    # Save DataFrame to CSV file
    df.to_csv(output_csv_path + '/dci_long.csv', index=False, sep='\t')