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
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"Dataset loaded successfully with encoding: {encoding}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def analyze_data(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    text_cols = [col for col in categorical_cols if data[col].str.len().mean() > 50]

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Text columns: {text_cols}")

    return numeric_cols, categorical_cols, text_cols

def generate_visualizations(data, numeric_cols):
    correlation = data[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved as correlation_heatmap.png")

def perform_clustering(data, numeric_cols, n_clusters=3):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        filtered_data = data[numeric_cols].dropna()  # Drop rows with NaNs for clustering
        clusters = kmeans.fit_predict(filtered_data)

        # Align cluster results back with original data length
        cluster_column = pd.Series(index=filtered_data.index, data=clusters)
        data['Cluster'] = cluster_column.reindex(data.index, fill_value=-1).astype(int)
        print(f"Clustering completed. Number of clusters: {n_clusters}")
        return data
    except Exception as e:
        print(f"Error performing clustering: {e}")
        sys.exit(1)

def query_llm(data):
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    api_key = os.getenv("AIPROXY_TOKEN")
    if not api_key:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

    summary = data.describe().to_dict()
    sample_rows = data.head(5).to_dict(orient='records')
    prompt = (
        f"### Dataset Summary:\n{summary}\n"
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

def save_readme(data, insights):
    with open("README.md", "w") as f:
        f.write(insights)
    print("README saved as README.md")

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_dataset(file_path)
    numeric_cols, categorical_cols, text_cols = analyze_data(data)

    if numeric_cols:
        generate_visualizations(data, numeric_cols)

    data = perform_clustering(data, numeric_cols)

    insights = query_llm(data)
    save_readme(data, insights)

if __name__ == "__main__":
    main()
