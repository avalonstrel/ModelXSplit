import os 
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "splitfdr",
    version = "0.0.1",
    author = "Temp",
    author_email = "Temp.com",
    description = ("The package of ModelX SplitKnockoff"),
    license = "BSD",
    keywords = "ModelX SplitKnockoff",
    packages=find_packages("src"),
    package_dir={"":"src"},
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)