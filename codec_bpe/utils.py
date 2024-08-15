from typing import List, Optional
import os

def get_codes_files(codes_path: str, num_files: Optional[int] = None) -> List[str]:
    codes_files = []
    for root, _, files in os.walk(codes_path):
        codes_files.extend([os.path.join(root, file) for file in files if file.endswith(".npy")])
    codes_files.sort()
    if num_files is not None:
        codes_files = codes_files[:num_files]
    return codes_files