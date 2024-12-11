# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "chardet",
#   "scikit-learn",
#   "requests"
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import chardet
import requests

def detect_encoding(file_path):
    """
    Detect file encoding for reading datasets.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        str: Detected file encoding.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def load_dataset(file_path):
    """
    Load the dataset using the detected encoding.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        encoding = detect_encoding(file_path)
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"Dataset loaded successfully with encoding: {encoding}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def analyze_data(data):
    """
    Analyze dataset to identify numeric, categorical, and text columns.
    Args:
        data (pd.DataFrame): Dataset to analyze.
    Returns:
        tuple: Lists of numeric, categorical, and text columns.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    text_cols = [col for col in categorical_cols if data[col].str.len().mean() > 50]

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Text columns: {text_cols}")

    return numeric_cols, categorical_cols, text_cols

def generate_visualizations(data, numeric_cols):
    """
    Generate visualizations for numeric columns in the dataset.
    Args:
        data (pd.DataFrame): Dataset to visualize.
        numeric_cols (list): List of numeric columns.
    """
    if numeric_cols:
        # Correlation heatmap
        correlation = data[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        print("Correlation heatmap saved as correlation_heatmap.png")

        # Additional visualizations
        for col in numeric_cols:
            plt.figure()
            sns.histplot(data[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(f"{col}_distribution.png")
            print(f"Histogram for {col} saved as {col}_distribution.png")

            plt.figure()
            sns.boxplot(y=data[col])
            plt.title(f"Boxplot of {col}")
            plt.ylabel(col)
            plt.savefig(f"{col}_boxplot.png")
            print(f"Boxplot for {col} saved as {col}_boxplot.png")

def perform_advanced_analysis(data, numeric_cols):
    """
    Perform advanced analyses like clustering and regression.
    Args:
        data (pd.DataFrame): Dataset to analyze.
        numeric_cols (list): List of numeric columns.
    """
    if numeric_cols:
        try:
            # Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(data[numeric_cols].dropna())
            data['Cluster'] = pd.Series(clusters, index=data.dropna().index)
            print("Clustering completed and added to dataset.")

            # Regression example
            if len(numeric_cols) > 1:
                X = data[numeric_cols[:-1]].dropna()
                y = data[numeric_cols[-1]].dropna()
                model = LinearRegression()
                model.fit(X, y)
                print(f"Regression model coefficients: {model.coef_}")
        except Exception as e:
            print(f"Advanced analysis error: {e}")

def query_llm(data):
    """
    Query the LLM for insights and README.md generation.
    Args:
        data (pd.DataFrame): Dataset to analyze.
    Returns:
        str: Generated insights from the LLM.
    """
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    api_key = os.getenv("AIPROXY_TOKEN")
    if not api_key:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

    summary = data.describe().to_dict()
    missing_values = data.isnull().sum().to_dict()
    sample_rows = data.head(5).to_dict(orient='records')
    prompt = (
        f"### Dataset Summary:\n{summary}\n"
        f"### Missing Values:\n{missing_values}\n"
        f"### Sample Rows:\n{sample_rows}\n"
        "Generate a detailed README.md with the following sections:\n"
        "1. Dataset Overview\n"
        "2. Key Insights\n"
        "3. Visualizations\n"
        "4. Suggested Further Analysis\n"
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error querying the LLM: {e}")
        sys.exit(1)

def save_readme(insights):
    """
    Save insights to README.md.
    Args:
        insights (str): Content to save in README.md.
    """
    with open("README.md", "w") as f:
        f.write(insights)
    print("README saved as README.md")

def main():
    """
    Main function to orchestrate the analysis, visualization, and LLM querying.
    """
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_dataset(file_path)
    numeric_cols, categorical_cols, text_cols = analyze_data(data)

    if numeric_cols:
        generate_visualizations(data, numeric_cols)
        perform_advanced_analysis(data, numeric_cols)

    insights = query_llm(data)
    save_readme(insights)

if __name__ == "__main__":
    main()
