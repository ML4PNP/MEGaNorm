from setuptools import setup, find_packages

# Load version from meganorm/_version.py
version = {}
with open("meganorm/_version.py") as f:
    exec(f.read(), version)

# Load dependencies from requirements.txt
def load_requirements(filename="requirements.txt"):
    with open(filename, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="meganorm",
    version=version["__version__"],
    description="A Python package for normative modeling on large-scale EEG and MEG data.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Seyed Mostafa Kia, Mohammad Zamanzadeh, Ymke Verduyn",
    license="GNU GPLv3",
    url="https://github.com/ML4PNP/MEGaNorm",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    include_package_data=True,
)
