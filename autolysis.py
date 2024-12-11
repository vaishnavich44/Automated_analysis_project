# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "chardet"
# ]
# ///

import os
import sys
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import chardet

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    # Load dataset with encoding detection
    try:
        # Detect file encoding
        with open(sys.argv[1], 'rb') as f:
            result = chardet.detect(f.read(10000))  # Detect encoding from a sample
            encoding = result['encoding']

        # Preprocess file content to handle encoding errors
        with open(sys.argv[1], 'r', encoding=encoding, errors='replace') as f:
            content = f.read()

        # Use pandas to read from the preprocessed content
        from io import StringIO
        data = pd.read_csv(StringIO(content))

        if data.empty:
            raise ValueError("The dataset is empty.")

        print(f"Dataset loaded successfully with encoding: {encoding}")
    except pd.errors.ParserError as pe:
        print(f"ParserError: {pe}")
        sys.exit(1)
    except UnicodeDecodeError as ude:
        print(f"UnicodeDecodeError: {ude}")
        sys.exit(1)
    except ValueError as ve:
        print(f"ValueError: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading the dataset: {e}")
        sys.exit(1)


    # Detect column types
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    text_columns = [col for col in categorical_columns if data[col].str.len().mean() > 50]

    print(f"Numeric columns: {numeric_columns}")
    print(f"Categorical columns: {categorical_columns}")
    print(f"Text columns: {text_columns}")

    # Perform analysis based on column types
    if numeric_columns:
        print("Performing correlation analysis for numeric columns...")
        correlation = data[numeric_columns].corr(numeric_only=True)
        print(correlation)
        generate_correlation_heatmap(correlation)

    if categorical_columns:
        print("Performing frequency analysis for categorical columns...")
        for col in categorical_columns:
            print(f"Top values for {col}:\n{data[col].value_counts().head()}")

    if text_columns:
        print("Displaying sample text for text columns...")
        for col in text_columns:
            print(f"Sample data from column {col}:")
            print(data[col].head())

    # Handle missing values
    print("Checking for missing values...")
    missing_values = data.isnull().sum()
    print(missing_values)
    if missing_values.sum() > 0:
        print("Handling missing values by dropping rows with missing data...")
        data = data.dropna()

    # Save narrative report and visualizations
    insights = generate_llm_insights(data, numeric_columns, categorical_columns, text_columns)
    save_narrative_report(data, insights)

def generate_correlation_heatmap(correlation):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved as correlation_heatmap.png")


def generate_llm_insights(data, numeric_columns, categorical_columns, text_columns):
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    api_key = os.getenv("AIPROXY_TOKEN")
    if not api_key:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

    # Prepare concise summary for LLM
    summary_stats = data.describe().to_dict()
    sample_rows = data.head(5).to_dict(orient='records')
    prompt = (
        f"Dataset Summary:\n"
        f"- Columns: {list(data.columns)}\n"
        f"- Summary Statistics: {summary_stats}\n"
        f"- Sample Rows: {sample_rows}\n"
        "Provide insights and suggest further analyses."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        insights = result['choices'][0]['message']['content'].strip()
        print("\nLLM Insights:\n")
        print(insights)
        return insights
    except requests.exceptions.RequestException as e:
        print(f"Error querying the LLM: {e}")
        sys.exit(1)

def save_narrative_report(data, insights):
    report = f"""
    # Analysis Report

    ## Dataset Overview
    {data.describe().to_string()}

    ## Key Insights
    {insights}
    """
    with open("README.md", "w") as f:
        f.write(report)
    print("Narrative report saved as README.md")

if __name__ == "__main__":
    main()
