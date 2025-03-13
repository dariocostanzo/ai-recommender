# Lesson 1: Getting Started with AI Engineering

## Introduction

Welcome to your personalized AI Engineering bootcamp! This series of lessons will take you from the fundamentals to building a complete AI-powered recommendation system. By learning through this project-based approach, you'll gain practical skills that are directly applicable to real-world AI engineering roles.

## What is AI Engineering?

AI Engineering is the discipline of building systems that can perform tasks that typically require human intelligence. As an AI Engineer, you'll:

- Design and implement machine learning models
- Work with data processing pipelines
- Deploy AI systems to production
- Maintain and improve AI applications over time

## Our Project: Content Recommendation System

Throughout this bootcamp, we'll build a content recommendation system that can:

1. Analyse text documents to understand their content
2. Find relationships between different pieces of content
3. Recommend related content based on user preferences
4. Provide personalised suggestions through a user-friendly interface

## Next Steps

In the next lesson, we'll:

1. Explore some sample data
2. Learn about text representations in AI
3. Build our first simple recommendation model

## Exercise

Before moving on, make sure you can:

1. Navigate to your project directory
2. Activate your virtual environment
3. Start a Jupyter notebook
4. Import basic libraries (numpy, pandas)

### Detailed Instructions

#### 1. Navigate to your project directory

Open a command prompt and navigate to your project directory:

```bash
cd c:\Users\[YOUR USERNAME]\projects\ai-recommender
```

#### 2. Create and activate your virtual environment

Create a new virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Your command prompt should now show `(venv)` at the beginning of the line.

#### 3. Install required packages

Install the necessary libraries:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter nltk spacy
```

#### 4. Start a Jupyter notebook

Launch Jupyter notebook:

```bash
jupyter notebook
```

This will open a browser window showing your project files. Navigate to the `notebooks` folder and create a new Python notebook by clicking on "New" → "Python 3".

#### 5. Import basic libraries

In your new notebook, add and run a cell with these imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Verify imports worked
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# Create a simple DataFrame
df = pd.DataFrame({
    'A': np.random.rand(5),
    'B': np.random.rand(5)
})
print("\nSample DataFrame:")
print(df)
```

If everything is set up correctly, you should see the library versions and a sample DataFrame printed out.

## Resources

- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Introduction to Machine Learning with Python](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)
