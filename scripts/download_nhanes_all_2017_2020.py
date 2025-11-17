#!/usr/bin/env python3
"""
Download all NHANES Data for 2017-2020 cycle (Demographics, Laboratory, Questionnaire, Examination).
"""

import urllib.request
import re
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests

BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"
CYCLE = "2017-2020"

# Components to download
COMPONENTS = {
    "Demographics": {
        "url": "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics&Cycle=2017-2020",
        "subdir": "demographics"
    },
    "Laboratory": {
        "url": "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2017-2020",
        "subdir": "laboratory"
    },
    "Questionnaire": {
        "url": "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2017-2020",
        "subdir": "questionnaire"
    },
    "Examination": {
        "url": "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2017-2020",
        "subdir": "examination"
    }
}

OUTPUT_BASE_DIR = Path(__file__).parent.parent / "data" / "nhanes" / CYCLE


class TqdmUpTo(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=output_path.name[:50]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {output_path.name}: {e}")
        return False


def extract_file_links_from_page(url):
    """Extract all XPT file links from the NHANES data page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        file_links = []
        
        # Find all links that point to XPT files
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Look for XPT file links - check both href and link text
            if '.XPT' in href.upper() or '.xpt' in href.lower() or 'XPT' in text.upper():
                # Convert relative URLs to absolute
                if href.startswith('http'):
                    file_url = href
                elif href.startswith('/'):
                    file_url = f"https://wwwn.cdc.gov{href}"
                elif href.startswith('../'):
                    # Handle relative paths like ../2017-2020/FILENAME.XPT
                    file_url = f"https://wwwn.cdc.gov/nchs/nhanes/{href.replace('../', '')}"
                else:
                    # Relative path - construct full URL
                    file_url = f"{BASE_URL}/{CYCLE}/{href}"
                
                # Extract filename
                file_name = href.split('/')[-1] if '/' in href else href
                if not file_name.endswith(('.XPT', '.xpt')):
                    # Try to extract from URL
                    if '.XPT' in file_url.upper():
                        file_name = file_url.split('/')[-1]
                
                if file_name.endswith(('.XPT', '.xpt')):
                    file_links.append((file_url, file_name))
        
        # Also search for file codes in table cells or data rows
        for cell in soup.find_all(['td', 'div', 'span']):
            text = cell.get_text(strip=True)
            # Look for patterns like "BMX_J.XPT" or "P_BMX_J"
            if '.XPT' in text.upper():
                # Try to extract filename
                match = re.search(r'([A-Z0-9_]+\.XPT)', text, re.IGNORECASE)
                if match:
                    filename = match.group(1).upper()
                    file_url = f"{BASE_URL}/{CYCLE}/{filename}"
                    file_links.append((file_url, filename))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for url, name in file_links:
            key = name.upper()
            if key not in seen:
                seen.add(key)
                unique_links.append((url, name))
        
        return unique_links
        
    except Exception as e:
        print(f"Error fetching page: {e}")
        return []


def download_component(component_name, component_info):
    """Download all files for a specific component."""
    url = component_info["url"]
    subdir = component_info["subdir"]
    output_dir = OUTPUT_BASE_DIR / subdir
    
    print(f"\n{'='*60}")
    print(f"Downloading {component_name} Data")
    print(f"{'='*60}")
    print(f"Source: {url}")
    print()
    
    # Extract file links from the page
    print("Fetching page to extract file links...")
    file_links = extract_file_links_from_page(url)
    
    if len(file_links) == 0:
        print(f"⚠ No files found for {component_name}. Skipping...")
        return 0, 0, 0
    
    print(f"\nFound {len(file_links)} {component_name} data files to download")
    print("-" * 60)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for file_url, filename in file_links:
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"⊙ Skipping {filename} (already exists)")
            skipped += 1
            continue
        
        print(f"\nDownloading {filename}...")
        if download_file(file_url, output_path):
            downloaded += 1
        else:
            failed += 1
    
    print(f"\n{component_name} Summary:")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Total files: {len(file_links)}")
    
    return downloaded, failed, skipped


def main():
    """Download all NHANES data components for 2017-2020."""
    print("="*60)
    print("NHANES 2017-2020 Complete Data Downloader")
    print("="*60)
    print("Downloading: Demographics, Laboratory, Questionnaire, Examination")
    print()
    
    total_downloaded = 0
    total_failed = 0
    total_skipped = 0
    
    # Download each component
    for component_name, component_info in COMPONENTS.items():
        downloaded, failed, skipped = download_component(component_name, component_info)
        total_downloaded += downloaded
        total_failed += failed
        total_skipped += skipped
    
    # Final summary
    print("\n" + "="*60)
    print("Overall Download Summary")
    print("="*60)
    print(f"  Successfully downloaded: {total_downloaded}")
    print(f"  Failed: {total_failed}")
    print(f"  Skipped (already exists): {total_skipped}")
    print(f"\nData saved to: {OUTPUT_BASE_DIR}")
    print("\nNote: NHANES files are in XPT format (SAS transport format).")
    print("  Use pandas.read_sas() to read these files:")
    print("  import pandas as pd")
    print("  df = pd.read_sas('file.xpt', format='xport')")
    print("="*60)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())
