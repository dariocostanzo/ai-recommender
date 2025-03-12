from setuptools import setup, find_packages

setup(
    name="ai-recommender",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "jupyter",
        "nltk",
        "spacy",
        "tensorflow",
        "torch"
    ],
)