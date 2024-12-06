from setuptools import find_packages, setup
from typing import List

# Correct variable name
hyphen_e_dot = "-e ."

# Function to read and process requirements
def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements from the specified file and returns a list of dependencies.
    Removes '-e .' if it exists in the file.
    """
    requirements = []
    try:
        with open(file_path, "r") as file_obj:
            requirements = file_obj.readlines()
            # Strip newline characters
            requirements = [req.strip() for req in requirements]
            # Remove '-e .' if present
            if hyphen_e_dot in requirements:
                requirements.remove(hyphen_e_dot)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found. Ensure it is in the same directory as setup.py.")
    return requirements

# Setup configuration
setup(
    name="Full_Project",
    version="0.0.1",
    author="Arlis",
    author_email="mt1221953@maths.iitd.ac.in",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
