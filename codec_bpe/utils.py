from typing import List, Optional, Union
import os

def get_codes_files(
    codes_path: str, 
    codes_filter: Optional[Union[str, List[str]]] = None, 
    num_files: Optional[int] = None,
) -> List[str]:
    if isinstance(codes_filter, str):
        codes_filter = [codes_filter]
    codes_files = []
    for root, _, files in os.walk(codes_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(".npy"):
                continue
            if codes_filter and not any([f in file_path for f in codes_filter]):
                continue
            codes_files.append(file_path)
    codes_files.sort()
    if num_files is not None:
        codes_files = codes_files[:num_files]
    return codes_files