from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="hypothesaes",
    version="0.0.1",
    packages=find_packages(),
    author="Rajiv Movva, Kenny Peng",
    author_email="rmovva@berkeley.edu, kennypeng@cs.cornell.edu",
    description="SAEs for hypothesis generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rmovva/HypotheSAEs",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)