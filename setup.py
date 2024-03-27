from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dynamic_graphs_with_pygame",
    version="1.0.3",
    description="A class for drawing dynamic graphs using Pygame.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Christos Botos",
    author_email="hcty02@gmail.com",
    url="https://github.com/ChrisBotos/Dynamic_Graphs_with_Pygame",
    packages=find_packages(),  # Use find_packages to automatically discover packages
    install_requires=["pygame", "numpy"],  # Add any dependencies here
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="pygame graphs dynamic visualization",
    
    project_urls={
        "LinkedIn": "https://www.linkedin.com/in/your_username/"
    }
)
