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
import chardet
import requests

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def load_dataset(file_path):
    try:
        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"Dataset loaded successfully with encoding: {encoding}")
        return data
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: Retrying with 'latin-1' encoding.")
        try:
            data = pd.read_csv(file_path, encoding="latin-1")
            print(f"Dataset loaded successfully with fallback encoding: latin-1")
            return data
        except Exception as e:
            print(f"Error loading dataset with fallback encoding: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def analyze_data(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    return numeric_cols, categorical_cols

def generate_visualizations(data, numeric_cols, categorical_cols):
    visualizations = []

    # Correlation Heatmap
    if numeric_cols:
        correlation = data[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()
        visualizations.append("correlation_heatmap.png")
        print("Saved: correlation_heatmap.png")

    # Histogram of First Numeric Column
    if numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[numeric_cols[0]], bins=30, kde=True)
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel("Frequency")
        plt.savefig(f"{numeric_cols[0]}_distribution.png")
        plt.close()
        visualizations.append(f"{numeric_cols[0]}_distribution.png")
        print(f"Saved: {numeric_cols[0]}_distribution.png")

    # Boxplot of Second Numeric Column
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=data[numeric_cols[1]])
        plt.title(f"Boxplot of {numeric_cols[1]}")
        plt.ylabel(numeric_cols[1])
        plt.savefig(f"{numeric_cols[1]}_boxplot.png")
        plt.close()
        visualizations.append(f"{numeric_cols[1]}_boxplot.png")
        print(f"Saved: {numeric_cols[1]}_boxplot.png")

    # Top Categories Bar Chart
    if categorical_cols:
        plt.figure(figsize=(10, 6))
        data[categorical_cols[0]].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top 10 Categories in {categorical_cols[0]}")
        plt.xlabel(categorical_cols[0])
        plt.ylabel("Frequency")
        plt.savefig(f"{categorical_cols[0]}_top10.png")
        plt.close()
        visualizations.append(f"{categorical_cols[0]}_top10.png")
        print(f"Saved: {categorical_cols[0]}_top10.png")

    return visualizations

def query_llm(data, numeric_cols, categorical_cols, visualizations):
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    api_key = os.getenv("AIPROXY_TOKEN")
    if not api_key:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

    summary_stats = data.describe(include="all").to_dict()
    missing_values = data.isnull().sum().to_dict()
    correlation_info = (
        data[numeric_cols].corr().to_dict() if numeric_cols else "No numeric columns available."
    )

    prompt = (
        "You are an expert data analyst. Generate a detailed README.md summarizing the analysis of this dataset. "
        "Focus on uncovering deep insights, drawing meaningful conclusions, and interpreting results in an engaging way. "
        "Tie the findings together into a cohesive narrative with a strong conclusion.\n\n"
        "### Dataset Overview\n"
        f"Numeric attributes: {numeric_cols}\n"
        f"Categorical attributes: {categorical_cols}\n\n"
        "### Insights & Visualizations\n"
        f"1. Correlation Heatmap\n"
        f"2. Distribution of {numeric_cols[0] if numeric_cols else 'N/A'}\n"
        f"3. Boxplot of {numeric_cols[1] if len(numeric_cols) > 1 else 'N/A'}\n"
        f"4. Top 10 Categories in {categorical_cols[0] if categorical_cols else 'N/A'}\n\n"
        "### Big Picture Conclusions\n"
        "Provide actionable insights and practical applications."
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
    with open("README.md", "w") as f:
        f.write(insights)
    print("Saved README.md")

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_dataset(file_path)
    numeric_cols, categorical_cols = analyze_data(data)

    visualizations = generate_visualizations(data, numeric_cols, categorical_cols)
    insights = query_llm(data, numeric_cols, categorical_cols, visualizations)
    save_readme(insights)

if __name__ == "__main__":
    main()
