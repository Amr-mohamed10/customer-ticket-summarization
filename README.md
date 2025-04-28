# Customer Ticket Summarization Project

This project uses a Streamlit application and a language model (GPT-4o-mini via Langchain) to summarize customer ticket histories based on product categories. It also includes LLM-as-a-judge evaluation and MLflow tracking.

## Features

* Loads ticket data from a CSV/TXT file.
* Filters tickets based on relevant service categories.
* Generates 5-section summaries (Initial Issue, Follow-ups, Developments, Later Incidents, Recent Events) for selected customers, grouped by product.
* Evaluates summary quality using an LLM-as-a-judge approach.
* Logs parameters, metrics, and evaluation results using MLflow.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Summarization
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Create a file named `.env` in the `Summarization` directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY='your_openai_api_key_here'
    ```

## Usage

1.  **Place your ticket data:** Ensure your ticket data file (e.g., `Ticket Data.txt`) is in the `Summarization` directory.
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
3.  Open your browser to the URL provided by Streamlit.
4.  Upload the ticket data file through the app interface.
5.  Select a customer number to generate and view summaries.

## MLflow

Evaluation results and run parameters are logged to MLflow. By default, this might create an `mlruns` directory locally. You can view the MLflow UI by running:
```bash
mlflow ui
