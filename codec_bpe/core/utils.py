from typing import List, Optional, Union
from argparse import Namespace
import json
import os

def get_codes_files(
    codes_path: str, 
    codes_filter: Optional[Union[str, List[str]]] = None, 
    num_files: Optional[int] = None,
) -> List[str]:
    return get_files(codes_path, ".npy", codes_filter, num_files)

def get_files(
    path: str, 
    extension: str,
    filter: Optional[Union[str, List[str]]] = None, 
    num_files: Optional[int] = None,
) -> List[str]:
    if isinstance(filter, str):
        filter = [filter]
    result_files = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(extension):
                continue
            if filter and not any([f in file_path for f in filter]):
                continue
            result_files.append(file_path)
    result_files.sort()
    if num_files is not None:
        result_files = result_files[:num_files]
    return result_files

def get_codec_info(codes_path: str) -> dict:
    codec_info_file = os.path.join(codes_path, "codec_info.json")
    if not os.path.exists(codec_info_file):
        return None
    with open(codec_info_file, "r") as f:
        codec_info = json.load(f)
    return codec_info

def update_args_from_codec_info(args: Namespace, codec_info: dict) -> Namespace:
    if codec_info is not None:
        if "num_codebooks" in args and args.num_codebooks is None:
            args.num_codebooks = codec_info["num_codebooks"]
        if "codebook_size" in args and args.codebook_size is None:
            args.codebook_size = codec_info["codebook_size"]
        if "codec_framerate" in args and args.codec_framerate is None:
            args.codec_framerate = codec_info["framerate"]
    return args