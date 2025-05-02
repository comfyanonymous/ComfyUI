#!/usr/bin/env python3
"""
Convert requirements.txt to pyproject.toml dependencies
"""

import re
import toml
from pathlib import Path

def parse_requirements(file_path):
    """Parse requirements file into a list of packages."""
    requirements = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                requirements.append(line)
    
    return requirements

def main():
    project_root = Path(__file__).parent
    pyproject_path = project_root / "pyproject.toml"
    
    # Parse requirements
    requirements = parse_requirements(project_root / "requirements.txt")
    
    try:
        advanced_requirements = parse_requirements(project_root / "requirements_advanced.txt")
    except FileNotFoundError:
        advanced_requirements = []
    
    # Load existing pyproject.toml if it exists
    if pyproject_path.exists():
        with open(pyproject_path, 'r') as f:
            pyproject_data = toml.load(f)
    else:
        # Create basic structure
        pyproject_data = {
            "build-system": {
                "requires": ["setuptools>=42", "wheel"],
                "build-backend": "setuptools.build_meta"
            },
            "project": {
                "name": "ComfyUI",
                "version": "0.1.0",
                "description": "The most powerful and modular diffusion model GUI, api and backend with a graph/nodes interface.",
                "readme": "README.md",
                "requires-python": ">=3.8",
                "dependencies": requirements
            }
        }
    
    # Update dependencies
    if "project" not in pyproject_data:
        pyproject_data["project"] = {}
    
    pyproject_data["project"]["dependencies"] = requirements
    
    # Add optional dependencies
    if advanced_requirements:
        if "optional-dependencies" not in pyproject_data["project"]:
            pyproject_data["project"]["optional-dependencies"] = {}
        
        pyproject_data["project"]["optional-dependencies"]["advanced"] = advanced_requirements
    
    # Save updated pyproject.toml
    with open(pyproject_path, 'w') as f:
        toml.dump(pyproject_data, f)
    
    print(f"Updated {pyproject_path} with dependencies from requirements files")

if __name__ == "__main__":
    main()
