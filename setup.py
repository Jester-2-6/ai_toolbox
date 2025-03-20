from setuptools import setup, find_packages

setup(
    name="ai_toolbox",                          # Replace with your package name
    version="0.1.0",                            # Follow semantic versioning
    author="Chinthana Wimalasuriya",
    author_email="chinthana.w@siu.edu",
    description="Toolbox for deep learning workflows",
    long_description=open("README.md").read(), # Use README.md for PyPI
    long_description_content_type="text/markdown",
    url="https://github.com/Jester-2-6/ai_toolbox",  # Repo URL
    packages=find_packages(),                  # Automatically find subpackages
    python_requires=">=3.7",                   # Minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
