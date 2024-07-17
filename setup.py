"""Setup script to make carbon-tracker-plants directly installable with pip."""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text()

setup(
    name='pudl_ct',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description='This repository is a collaboration between CarbonTracker and '
    'Catalyst Cooperative.',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/catalyst-cooperative/carbon-tracker-plants/',
    license="MIT",
    version='0.0.1',
    install_requires=[
        "catalystcoop.pudl",
    ],
    python_requires=">=3.8,<3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    author="Catalyst Cooperative",
    author_email="pudl@catalyst.coop",
    maintainer="Christina Gosnell",
    maintainer_email="cgosnell@catalyst.coop",
    keywords=['coal', 'gas', 'eia', 'ferc1', 'cems', 'ampd', 'power plants']
)
