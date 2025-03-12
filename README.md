# AI Recommender System

This project implements a content recommendation system using AI techniques. It analyzes text documents to understand their content, finds relationships between different pieces of content, and recommends related content based on user preferences.

## Project Structure

```

ai-recommender/
├── data/ # For storing datasets
├── models/ # For saving trained models
├── notebooks/ # For exploratory data analysis
├── src/ # Source code for our application
└── docs/ # Documentation and lessons
├── lessons/ # Step-by-step tutorials
├── resources/ # Additional learning materials
└── exercises/ # Practice problems

```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Environment Setup

1. Clone the repository:

```

git clone https://github.com/dariocostanzo/ai-recommender.git
cd ai-recommender

```

2. Create a virtual environment:

```

python -m venv venv

```

3. Activate the virtual environment:

- Windows:
  ```
  venv\Scripts\activate
  ```
- macOS/Linux:
  ```
  source venv/bin/activate
  ```

4. Install dependencies:

```

pip install -r requirements.txt

```

### Running Jupyter Notebooks

1. Ensure your virtual environment is activated (you should see `(venv)` at the beginning of your command prompt)

2. Start Jupyter Notebook:

```

jupyter notebook

```

3. Your browser will open automatically. Navigate to the `notebooks` directory to access the project notebooks.

4. To verify your environment is set up correctly, run the first notebook which contains basic imports and tests.

### Running the Application

The application components can be run individually for testing:

```

python src/test_text_processor.py

```

## Features

- Text document analysis
- Content relationship mapping
- Personalized recommendations
- User-friendly interface

## Development

To contribute to this project:

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

## License

[MIT License](LICENSE)
