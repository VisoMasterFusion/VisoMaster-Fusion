import hashlib

# --- Constants ---

# The buffer size (in bytes) for reading files in chunks.
# Reading large files in smaller chunks is memory-efficient, as it avoids
# loading the entire file into memory at once. 128KB is a common and effective size.
BUF_SIZE = 131072  # 128kb chunks

def get_file_hash(file_path: str) -> str:
    """
    Calculates the SHA256 hash of a file.

    It reads the file in chunks to efficiently handle large files without
    consuming excessive memory.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The hexadecimal representation of the SHA256 hash.
    """
    # Initialize the SHA256 hash object.
    hash_sha256 = hashlib.sha256()

    # Open the file in binary read mode ('rb').
    with open(file_path, 'rb') as f:
        while True:
            # Read a chunk of the file.
            data = f.read(BUF_SIZE)
            if not data:
                # End of file has been reached.
                break
            # Update the hash object with the chunk.
            hash_sha256.update(data)

    return hash_sha256.hexdigest()

def write_hash_to_file(hash_value: str, hash_file_path: str) -> None:
    """
    Writes a given hash string to a specified text file.

    Args:
        hash_value (str): The hash string to write.
        hash_file_path (str): The path to the output file.
    """
    with open(hash_file_path, 'w') as hash_file:
        hash_file.write(hash_value)

def get_hash_from_hash_file(hash_file_path: str) -> str:
    """
    Reads a hash string from a specified text file.

    Args:
        hash_file_path (str): The path to the hash file.

    Returns:
        str: The hash string, with leading/trailing whitespace removed.
    """
    with open(hash_file_path, 'r') as hash_file:
        hash_sha256 = hash_file.read().strip()
        return hash_sha256
    
def check_file_integrity(file_path: str, correct_hash: str) -> bool:
    """
    Verifies the integrity of a file by comparing its actual hash
    with an expected hash.

    Args:
        file_path (str): The path to the file to check.
        correct_hash (str): The expected SHA256 hash.

    Returns:
        bool: True if the file's hash matches the expected hash, False otherwise.
    """
    actual_hash = get_file_hash(file_path)
    return actual_hash == correct_hash