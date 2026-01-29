# README Link Checker
#
# This script checks the online status of links in a README.md file. It extracts all URLs from the README file
# and sends a HEAD request to each URL to determine if the link is online or not.
#
#     python check_readme_links.py path/to/README.md
#
# Author: Brandon Himpfen
# Website: himpfen.xyz

import argparse
import re
import requests
from pathlib import Path

def check_links(file_path):
    contents = Path(file_path).read_text()

    # Extract all URLs from the README file
    urls = re.findall(r'\[.*\]\((http[s]?://.*?)\)', contents)

    for url in urls:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            if response.status_code == 200:
                print(f"Link {url} is online.")
            else:
                print(f"Link {url} returned status code {response.status_code}.")
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while checking link {url}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Check links in a README file.")
    parser.add_argument("readme", nargs="?", default="README.md", help="Path to README.md")
    args = parser.parse_args()
    check_links(args.readme)


if __name__ == "__main__":
    main()
