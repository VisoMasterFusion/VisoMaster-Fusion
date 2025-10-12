import requests
from pathlib import Path
import os
from tqdm import tqdm
from app.helpers.integrity_checker import check_file_integrity

# --- Constants ---
# Use named constants for values that appear in the code. This makes the code
# more readable and easier to modify.
DOWNLOAD_CHUNK_SIZE = 1024      # Size of download chunks in bytes.
MAX_DOWNLOAD_ATTEMPTS = 3       # Number of retries for a failed download.
REQUEST_TIMEOUT = 10            # Timeout for network requests in seconds.

def download_file(model_name: str, file_path: str, correct_hash: str, url: str) -> bool:
    """
    Downloads a model file from a given URL, verifies its integrity using a hash,
    and handles retries on failure.

    This function is robust: it first checks if a valid file already exists.
    If the download fails, it will retry up to MAX_DOWNLOAD_ATTEMPTS times.
    A progress bar is displayed during the download.

    Args:
        model_name (str): The human-readable name of the model (e.g., "RetinaFace").
        file_path (str): The local path where the file should be saved.
        correct_hash (str): The expected SHA256 hash for integrity verification.
        url (str): The URL from which to download the file.

    Returns:
        bool: True if the file was successfully downloaded and verified, False otherwise.
    """
    
    # First, check if the file already exists and has the correct integrity.
    # This avoids re-downloading large files unnecessarily.
    if Path(file_path).is_file():
        if check_file_integrity(file_path, correct_hash):
            print(f"\nSkipping '{model_name}': file already exists and is valid.")
            return True
        else:
            # If the file exists but is corrupt, remove it before downloading again.
            print(f"\nFile '{file_path}' exists but is corrupt. Re-downloading...")
            os.remove(file_path)

    print(f"\nDownloading '{model_name}' from {url}")

    # Use a for loop for a fixed number of retries. It's cleaner than a while loop with a counter.
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            # Using a 'with' statement for the request ensures that the connection
            # is properly closed even if errors occur.
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                # Raise an exception for bad status codes (like 404 Not Found or 500 Server Error).
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                
                # The tqdm progress bar provides a great user experience for long downloads.
                # The 'desc' parameter adds a useful description.
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=model_name) as progress_bar:
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                            if chunk: # filter out keep-alive new chunks
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                
                # After a successful download, verify the file's integrity.
                if check_file_integrity(file_path, correct_hash):
                    print(f"\nSuccessfully downloaded and verified '{model_name}'.")
                    print(f"File saved at: {file_path}")
                    return True # Exit the function on success.
                else:
                    print(f"\nDownload complete, but integrity check failed (Attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}).")
                    os.remove(file_path) # Clean up the corrupt file before retrying.

        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred during download (Attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}): {e}")
        
        # This message will be shown if the loop continues to the next attempt.
        if attempt < MAX_DOWNLOAD_ATTEMPTS:
            print("Retrying...")

    # This message is displayed only if all attempts have failed.
    print(f"\nFailed to download '{model_name}' after {MAX_DOWNLOAD_ATTEMPTS} attempts.")
    return False