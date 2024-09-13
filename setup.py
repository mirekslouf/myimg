import setuptools

def get_version():
    with(open("src/myimg/__init__.py", "r")) as fh:
        for line in fh:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")
                
def get_long_description():
    with open("README.md", "r") as fh: description = fh.read()
    return(description)
    
setuptools.setup(
    name="myimg",
    version=get_version(),
    author="Mirek Slouf",
    author_email="mirek.slouf@gmail.com",
    description="Batch processing of micrographs",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mirekslouf/myimg/",
    project_urls={
        "Documentation": "https://mirekslouf.github.io/myimg/"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    license='MIT',
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",     # 3.6 for f-strings, 3.7 for dataclasses
    include_package_data=True)
