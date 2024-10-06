from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dynamic_graphs_with_pygame",
    version="1.2.0",
    description="A class for drawing dynamic graphs using Pygame.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Christos Botos",
    author_email="hcty02@gmail.com",
    url="https://github.com/ChrisBotos/Dynamic_Graphs_with_Pygame",
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        "pygame>=2.0.0",  # Specifying minimum versions can help avoid compatibility issues
        "numpy>=1.18.0"
    ],  # List your dependencies here
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="pygame graphs dynamic visualization, pygame visualization, dynamic graphs",
    project_urls={
        "Source": "https://github.com/ChrisBotos/Dynamic_Graphs_with_Pygame",
        "PyPI": "https://pypi.org/project/dynamic-graphs-with-pygame/",
        "Documentation": "https://github.com/ChrisBotos/Dynamic_Graphs_with_Pygame/blob/main/README.md",
        "LinkedIn": "https://www.linkedin.com/in/christos-botos-2369hcty3396",
    },
    python_requires='>=3.7',  # Specify the Python version compatibility
)
