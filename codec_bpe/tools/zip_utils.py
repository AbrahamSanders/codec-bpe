import os
import zipfile
import requests
import shutil
from urllib.parse import urlparse

def download_and_extract_zip(url: str, extract_to: str = ".", delete_zip: bool = True) -> None:
    """
    Downloads a zip file from the given URL, extracts it to the specified path,
    and optionally deletes the original zip file.
    
    Args:
        url: The URL of the zip file to download
        extract_to: The directory to extract the zip contents to
        delete_zip: Whether to delete the zip file after extraction (default: True)
    
    Raises:
        requests.RequestException: If there's an error downloading the file
        zipfile.BadZipFile: If the downloaded file is not a valid zip file
        IOError: If there's an error in file operations
    """
    # Create the target directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Get the filename from the URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename.endswith('.zip'):
        filename += '.zip'
    
    # Full path for the downloaded zip file
    zip_path = os.path.join(extract_to, filename)
    
    try:
        # Download the file
        print(f"Downloading {url} to {zip_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Write the content to a file
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        # Extract the zip file
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Delete the zip file if requested
        if delete_zip:
            print(f"Deleting {zip_path}...")
            os.remove(zip_path)
            
        print("Download and extraction completed successfully")
            
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        raise
    except zipfile.BadZipFile:
        print(f"The downloaded file is not a valid zip file")
        # Clean up the invalid file
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
