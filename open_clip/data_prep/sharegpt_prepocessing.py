import json
import csv
from tqdm import tqdm
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Transform ShareGPT data to CSV format')
    parser.add_argument('--data-path', type=str, help='Path to ShareGPT4V directory', default='/ShareGPT4V')
    parser.add_argument('--json-name', type=str, help='Path to the input JSON file', default="share-captioner_coco_lcs_sam_1246k_1107")
    parser.add_argument('--split', type=str, help='dataset split', default="train", choices=["train", "val"])
    args = parser.parse_args()
    return args


def get_csv(args):
    # Path to the input JSON file
    json_file_path = f'{args.data_path}/{args.json_name}.json'

    # Path to the output CSV file
    csv_file_path = f'{args.data_path}/{args.json_name}.csv'

    # Path to the images
    images_path = f'{args.data_path}/data/'

    # Read the JSON data
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

        # Open the CSV file for writing
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['filepath', 'title'])

            # Write each item from the JSON data
            for item in tqdm(data, desc='Writing data to CSV'):
                image_name = f"{images_path}{item['image']}" 
                caption = item['conversations'][1]['value'].strip().replace('\n', ' ')
                    
                # Write the row to the CSV file
                csv_writer.writerow([image_name, caption])

                short_caption = f"{caption.split('.')[0]}." 
                csv_writer.writerow([image_name, short_caption])

        print(f"Data has been transformed and written to {csv_file_path}")


if __name__ == '__main__':  
    # Parse the command line arguments
    args = get_parser()
    get_csv()
