import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Bootmedian",
    version = "1.0.0",
    author = "Alejandro S. Borlaff",
    author_email = "a.s.borlaff@nasa.gov",
    description = ("A software to estimate medians through Bootstrapping"),
    license = "BSD",
    keywords = "Statistics / Bootstrapping",
    url = "https://github.com/Borlaff/bootmedian",
    packages=setuptools.find_packages(where=".", exclude=()),
    package_data={'': ['*.mplstyle', '*.csv', '*.txt']},
    install_requires=[
      'numpy==1.26.4', 'multiprocess', 'bottleneck', 'pandas', 'tqdm', 'astropy', 'matplotlib', 'miniutils'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
