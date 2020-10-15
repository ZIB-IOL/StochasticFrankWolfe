import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name            = "frankwolfe-IOL",
    version         = "1.0.0",
    author          = "IOL Lab @ TUB/ZIB",
    author_email    = "spiegel@zib.de",
    description     = "Pytorch and Tensorflow implementations of Stochastic Franl-Wolfe methods",
    url             = "github.com/ZIB-IOL/StochasticFrankWolfe",
    packages        = setuptools.find_packages(),
    classifiers     = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
)
