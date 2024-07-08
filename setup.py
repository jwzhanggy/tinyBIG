from setuptools import setup, find_packages

__version__ = '0.1.0.post9'

requirements = [
    "torch==2.2.2",
    "numpy==1.26.3",
    "pyyaml==6.0.1",
    "scipy==1.13.1",
    "tqdm==4.66.4",
    "torchvision==0.17.2",
    "torchtext==0.17.2",
    "scikit-learn==1.5.1",
    "matplotlib==3.9.1",
]

test_requirements = [
    "pytest>=6"
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
long_description_type = "text/markdown"

setup(
    name="tinybig",
    version=__version__,
    author="Jiawei Zhang",
    author_email="jiawei@ifmlab.org",

    description="Deep function learning with reconciled polynomial network",
    long_description=long_description,
    long_description_content_type=long_description_type,
    keywords=[
        "tinybig",
        "rpn",
        "deep function learning",
        "expansion function",
        "reconciliation function",
        "remainder function",
        "reconciled polynomial network",
    ],

    url="https://www.tinybig.org",
    download_url="https://github.com/jwzhanggy/tinyBIG",
    packages=find_packages(include=["tinybig", "tinybig.*"]),

    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires='>=3.8',
    install_requires=requirements,

    test_suite="tests",
    tests_require=test_requirements,
)


# python setup.py sdist bdist_wheel
# to testpypi: twine upload --repository testpypi dist/*
# to pypi: twine upload dist/*