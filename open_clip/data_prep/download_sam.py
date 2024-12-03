import os
import requests
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(filename, 'wb') as file, tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, desc=filename, ncols=80, leave=False) as progress_bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return None

def process_tsv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            fields = line.strip().split('\t')
            # Extract the numeric part of the file name
            match = re.search(r'\d+', fields[0])
            if match:
                numeric_part = int(match.group())
                data.append([numeric_part, fields[0], fields[1]])
            else:
                print(f"Could not extract numeric part from file name: {fields[0]}")

    data.sort(key=lambda x: x[0])  # Sort by the numeric part of the file name (ascending)
    data = data[:50]  # Limit to the first 50 files

    selected_filenames = ["sa_000002.tar", 
                "sa_000004.tar", 
                "sa_000006.tar", 
                "sa_000008.tar", 
                "sa_000009.tar", 
                "sa_000011.tar", 
                "sa_000013.tar", 
                "sa_000014.tar", 
                "sa_000015.tar", 
                "sa_000017.tar", 
                "sa_000018.tar", 
                "sa_000019.tar", 
                "sa_000022.tar",
                "sa_000024.tar",
                "sa_000025.tar",
                "sa_000027.tar",
                "sa_000032.tar",
                "sa_000033.tar",
                "sa_000034.tar",
                "sa_000035.tar",
                "sa_000036.tar",
                "sa_000038.tar",
                "sa_000039.tar",
                "sa_000040.tar",
                "sa_000041.tar",
                "sa_000042.tar",
                "sa_000048.tar",
                "sa_000049.tar",]
    
    data = [row for row in data if row[1] in selected_filenames]


    # Download the first 50 files in parallel
    with ProcessPoolExecutor() as executor:
        filenames = [filename for filename in executor.map(download_file, [row[2] for row in data], [row[1] for row in data[:1]]) if filename]

    print(f"Downloaded the first 50 files: {', '.join(filenames)}")

if __name__ == "__main__":
    process_tsv_file('/ShareGPT4V/data/sam.tsv')