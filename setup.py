import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quantimage-melampus", # Replace with your own username
    version="0.0.2",
    author="Orfeas Aidonopoulos",
    author_email="orfeas.aidonopoulos@hevs.ch",
    description="Melampus: Machine learning package for Imagine infrastracture.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/orfi2017/quantimage-melampus/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
