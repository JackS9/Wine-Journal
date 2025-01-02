import requests
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import pprint

# Step 1: Fetch the webpage
url = "https://www.vivino.com/US/en/fr-cormeil-figeac-saint-emilion-grand-cru/w/2160185?srsltid=AfmBOoqWAhiuLx-wmP8FsecBnaRACKwlrCL-Wk0hmkTQ5O5wYMO_wtYF"
response = requests.get(url)
html_content = response.text

# Step 2: Parse the HTML content with BeautifulSoup
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Step 3: Find all <script> tags
script_tags = soup.find_all("script")
prefix = ""
# Step 4: Search for a specific pattern in each script tag
target_pattern = r"PageInformation"  # Regex to find any occurrence of 'VariableEnding'
def iterate_nested_json(json_obj):
    global prefix
    old_prefix = prefix
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, dict) or isinstance(value, list):
                prefix = prefix + f"[{key}]"
                iterate_nested_json(value)
                prefix = old_prefix
            else:
                if prefix.startswith("[vintage]"):
                    print(f"{prefix}[{key}]: {value}")
    elif isinstance(json_obj, list):
        for index, value in enumerate(json_obj):
            if isinstance(value, dict) or isinstance(value, list):
                prefix = prefix + f"[{index}]"
                iterate_nested_json(value)
                prefix = old_prefix
            else:
                if prefix.startswith("[vintage]"):
                    print(f"{prefix}[{index}]{value}")
    else:
        if prefix.startswith("[vintage]"):
            print(f"{prefix}{value}")



for script_tag in script_tags:
    if script_tag.string:  # Ensure the script tag has content
        match = re.search(target_pattern, script_tag.string)
        if match:
            script_content = script_tag.string
            # Regex to extract JSON data assigned to 'targetVariable'
            pattern = r"(\w*PageInformation)\s*=\s*(\{.*?\});"
            match = re.search(pattern, script_content, re.DOTALL)

            if match:
                json_data = match.group(2)  # Extract the JSON string
                #pprint.pprint(json_data)
                parsed_data = json.loads(json_data)  # Convert to Python dictionary
                iterate_nested_json(parsed_data)
                #print(parsed_data)
                break
            else:
                print("Target variable not found in the script.")
        else:
            print("Target pattern not found in the script.")
    else:
        print("No string found in this script.")
