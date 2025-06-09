import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description. It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Function to read the requirements from requirements.txt
def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="qml4omics",  # Replace with your project's name
    version="0.0.1",         # Start with a small version number and increment as needed
    author="Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida",      # Replace with your name
    author_email="your.email@example.com", # Replace with your email
    description=("This package automates the use the of classical and quantum machine learning as well as the generation of dataset complexity measures, and the subsequent correlations between these dataset complexity measures and "
    "predictive model performance.  It allows for the profiling of omics datasets, including genomics, proteomics, metabolomics, and other healthcare and life science (HCLS data)"
    "to identify the most suitable machine learning techniques for analysis.  The package includes an oracle (QSage), which trains on these correlations, allowing users to predict the performance of machine learning models on new datasets based on their complexity measures, before hand."
    "This can then be used to guide a tailored selection of models for the main profiler function to run. "), # Short description of your project
    license="Apache 2.0",           # Choose an appropriate license (e.g., MIT, Apache 2.0, GPLv3)
    keywords="machine learning, omics, qml, oracle, profiler", # Keywords for your project
    url="https://github.com/IBM/qml4omics/tree/", # Replace with your project's URL (e.g., GitHub repo)
    packages=find_packages(), # Automatically find all packages in your project
    long_description=read('README.md') if os.path.exists('README.md') else '', # Reads from README.md if present
    long_description_content_type='text/markdown', # Set this if your README is in Markdown
    classifiers=[
        "Development Status :: 3 - Alpha", # Choose appropriate development status
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache 2.0",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    
        "Operating System :: OS Independent",
    ],
    install_requires=read_requirements(), # This pulls dependencies from requirements.txt
    python_requires='>=3.7', # Specify the minimum Python version required
    # If you have non-code files that need to be included in your package,
    # uncomment and configure package_data or include_package_data
    # package_data={
    #     'your_package_name': ['data/*.txt', 'templates/*.html'],
    # },
    # include_package_data=True, # Set to True to include data files specified in MANIFEST.in
    # entry_points={
    #     'console_scripts': [
    #         'my_project_command=my_project_name.cli:main', # Example: if you have a command-line script
    #     ],
    # },
)