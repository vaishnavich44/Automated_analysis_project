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
from sklearn.decomposition import PCA
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
        return data
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="latin-1")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def analyze_data(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

def detect_outliers(data, numeric_cols):
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_summary[col] = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
    return outlier_summary

def perform_clustering(data, numeric_cols):
    try:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(data[numeric_cols].dropna())
        data['Cluster'] = clusters
    except Exception as e:
        print(f"Clustering failed: {e}")

def apply_pca(data, numeric_cols):
    try:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data[numeric_cols].dropna())
        data['PCA1'], data['PCA2'] = pca_result[:, 0], pca_result[:, 1]
    except Exception as e:
        print(f"PCA transformation failed: {e}")

def generate_visualizations(data, numeric_cols, categorical_cols):
    visualizations = []

    if numeric_cols:
        correlation = data[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()
        visualizations.append("correlation_heatmap.png")

        plt.figure(figsize=(8, 6))
        sns.histplot(data[numeric_cols[0]], bins=30, kde=True)
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.savefig(f"{numeric_cols[0]}_distribution.png")
        plt.close()
        visualizations.append(f"{numeric_cols[0]}_distribution.png")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data)
        plt.title("PCA Clustering Scatterplot")
        plt.savefig("pca_clustering_scatterplot.png")
        plt.close()
        visualizations.append("pca_clustering_scatterplot.png")

    if categorical_cols:
        top_categories = data[categorical_cols[0]].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
        plt.title(f"Top 10 Categories in {categorical_cols[0]}")
        plt.savefig(f"{categorical_cols[0]}_top10.png")
        plt.close()
        visualizations.append(f"{categorical_cols[0]}_top10.png")

    return visualizations

def query_llm(prompt):
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    api_key = os.getenv("AIPROXY_TOKEN")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error querying the LLM: {e}")
        sys.exit(1)

def generate_dynamic_prompt(data, numeric_cols, categorical_cols, outliers, visualizations):
    prompt = (
        "You are an expert data analyst. Summarize the analysis of this dataset.\n\n"
        "### Dataset Overview\n"
        f"Numeric Columns: {numeric_cols}\n"
        f"Categorical Columns: {categorical_cols}\n\n"
        "### Insights\n"
        f"1. Outliers detected: {outliers}\n"
        f"2. Correlations identified in numeric columns.\n\n"
        "### Visualizations\n"
        f"1. Correlation Heatmap\n"
        f"2. Distribution of {numeric_cols[0] if numeric_cols else 'N/A'}\n"
        f"3. PCA Clustering Scatterplot\n"
        f"4. Top 10 Categories in {categorical_cols[0] if categorical_cols else 'N/A'}\n\n"
        "### Recommendations\n"
        "Provide recommendations based on the findings."
    )
    return prompt

def save_readme(insights):
    with open("README.md", "w") as f:
        f.write(insights)

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_dataset(file_path)
    numeric_cols, categorical_cols = analyze_data(data)

    outliers = detect_outliers(data, numeric_cols)
    perform_clustering(data, numeric_cols)
    apply_pca(data, numeric_cols)
    visualizations = generate_visualizations(data, numeric_cols, categorical_cols)

    prompt = generate_dynamic_prompt(data, numeric_cols, categorical_cols, outliers, visualizations)
    insights = query_llm(prompt)
    save_readme(insights)

if __name__ == "__main__":
    main()
