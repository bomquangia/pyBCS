import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyBCS-bioturing",
    version="0.3.1",
    author="BioTuring",
    author_email="support@bioturing.com",
    description="Create BioTuring Compressed Study (bcs) file",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.bioturing.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "scanpy>=1.6.0",
        "anndata>=0.7.5",
        "loompy>=3.0.6"
    ]
)
