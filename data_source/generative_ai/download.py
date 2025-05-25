import json
import os
import wget
import re

# Load the JSON file into file_links
json_file = './data_source/data_acquisition/arxiv_large_language_model.json'
with open(json_file, 'r', encoding='utf-8') as f:
    file_links = json.load(f)

# Define your target directory and make sure it exists
output_dir = './data_source/generative_ai'
os.makedirs(output_dir, exist_ok=True)

# Helper to make a filesystem-safe filename
def sanitize_filename(title: str) -> str:
    # keep letters, numbers, spaces, dashes and underscores; replace all else with '_'
    return re.sub(r'[^\w\s\-]', '_', title).strip()

# Check for existence in the output folder
def is_exist(file_link):
    fname = sanitize_filename(file_link['title']) + '.pdf'
    return os.path.exists(os.path.join(output_dir, fname))

# Download any missing PDFs into output_dir
for file_link in file_links:
    title   = file_link['title']
    pdf_url = file_link['url']
    fname   = sanitize_filename(title) + '.pdf'
    out_path = os.path.join(output_dir, fname)
    
    if not os.path.exists(out_path):
        print(f"Downloading “{title}” to {out_path} …")
        try:
            wget.download(pdf_url, out=out_path)
            print()  # newline after wget’s progress bar
        except Exception as e:
            print(f"\n Failed to download “{title}”: {e}")
    else:
        print(f"Already have: {out_path}")
