import io
import os

from setuptools import find_packages, setup

for line in open("codec_bpe/__init__.py"):
    line = line.strip()
    if "__version__" in line:
        context = {}
        exec(line, context)
        VERSION = context["__version__"]


def read(*paths, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8")) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]


setup(
    name="codec-bpe",
    version=VERSION,
    author="Abraham Sanders",
    author_email="abraham.sanders@gmail.com",
    description="Implementation of Acoustic BPE (Shen et al., 2024), extended for RVQ-based Neural Audio Codecs",
    url="https://github.com/AbrahamSanders/codec-bpe",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "funcodec": read_requirements("requirements_funcodec.txt"),
        "xcodec2": read_requirements("requirements_xcodec2.txt"),
        "wavtokenizer": read_requirements("requirements_wavtokenizer.txt"),
        "simvq": read_requirements("requirements_simvq.txt"),
        "magicodec": read_requirements("requirements_magicodec.txt"),
    },
)