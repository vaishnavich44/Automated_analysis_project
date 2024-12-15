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
    """
    Detects the encoding of the file to handle various character sets.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        str: Detected encoding.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def load_dataset(file_path):
    """
    Loads the dataset from the given file path, handling encoding issues.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
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
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the file path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def analyze_data(data):
    """
    Analyzes the dataset to identify numeric and categorical columns.
    Args:
        data (pd.DataFrame): The dataset to analyze.
    Returns:
        tuple: Lists of numeric and categorical columns.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    return numeric_cols, categorical_cols

def detect_outliers(data, numeric_cols):
    """
    Detects outliers in numeric columns using the IQR method.
    Args:
        data (pd.DataFrame): The dataset to analyze.
        numeric_cols (list): List of numeric column names.
    Returns:
        dict: A summary of outliers per numeric column.
    """
    outlier_summary = {}
    if not numeric_cols:
        print("No numeric columns available for outlier detection.")
        return outlier_summary
    
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_summary[col] = len(outliers)
    print(f"Outlier summary: {outlier_summary}")
    return outlier_summary

def perform_clustering(data, numeric_cols):
    if not numeric_cols:
        print("Skipping clustering: No numeric columns available.")
        return

    try:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(data[numeric_cols].dropna())
        data['Cluster'] = clusters
        print("Clustering completed. Cluster labels added to the dataset.")
    except Exception as e:
        print(f"Clustering failed: {e}")

def apply_pca(data, numeric_cols):
    if not numeric_cols:
        print("Skipping PCA: No numeric columns available.")
        return

    try:
        pca = PCA(n_components=2)
        # Dropping rows with missing values
        data_numeric = data[numeric_cols].dropna()
        pca_result = pca.fit_transform(data_numeric)
        
        # Create a DataFrame with PCA results and align it with the original data
        pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'], index=data_numeric.index)
        data['PCA1'] = pca_df['PCA1']
        data['PCA2'] = pca_df['PCA2']
        print("PCA applied. PCA results added to the dataset.")
    except Exception as e:
        print(f"PCA failed: {e}")

def generate_visualizations(data, numeric_cols, categorical_cols):
    """
    Generates visualizations based on the dataset.
    Args:
        data (pd.DataFrame): The dataset to visualize.
        numeric_cols (list): List of numeric column names.
        categorical_cols (list): List of categorical column names.
    Returns:
        list: File paths of the generated visualizations.
    """
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

        # Histogram for the first numeric column
        plt.figure(figsize=(8, 6))
        sns.histplot(data[numeric_cols[0]], bins=30, kde=True)
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.savefig(f"{numeric_cols[0]}_distribution.png")
        plt.close()
        visualizations.append(f"{numeric_cols[0]}_distribution.png")

    # Scatter plot for PCA clustering
    if 'PCA1' in data.columns and 'PCA2' in data.columns and 'Cluster' in data.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data)
        plt.title("PCA Clustering Scatterplot")
        plt.savefig("pca_clustering_scatterplot.png")
        plt.close()
        visualizations.append("pca_clustering_scatterplot.png")
        print("Saved: pca_clustering_scatterplot.png")
    else:
        print("Skipping PCA Clustering Scatterplot: Required columns ('PCA1', 'PCA2', 'Cluster') are missing.")

    # Top categories bar chart
    if categorical_cols:
        plt.figure(figsize=(10, 6))
        top_categories = data[categorical_cols[0]].value_counts().head(10)
        sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
        plt.title(f"Top 10 Categories in {categorical_cols[0]}")
        plt.savefig(f"{categorical_cols[0]}_top10.png")
        plt.close()
        visualizations.append(f"{categorical_cols[0]}_top10.png")
        print(f"Saved: {categorical_cols[0]}_top10.png")

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
        "You are an expert data analyst. Generate a detailed README.md summarizing the analysis of this dataset. "
        "Focus on uncovering deep insights, drawing meaningful conclusions, and interpreting results in an engaging way. "
        "Tie the findings together into a cohesive narrative with a strong conclusion. "
        "Include the visualizations directly into the narrative for clarity.\n\n"
        "### Dataset Overview\n"
        f"Numeric Columns: {numeric_cols}\n"
        f"Categorical Columns: {categorical_cols}\n\n"
        "### Key Insights\n"
        f"1. Correlation insights: {data[numeric_cols].corr().to_dict() if numeric_cols else 'No numeric columns available.'}\n"
        f"2. Outlier summary: {outliers}\n"
        "3. Missing values analysis: Discuss potential impacts and resolution strategies.\n\n"
        "### Visualizations\n"
    )

    for viz in visualizations:
        prompt += f"- ![{viz}]({viz})\n"

    prompt += (
        "\n### Practical Applications\n"
        "Explain how the findings can inform decisions in relevant domains.\n\n"
        "### Big Picture Conclusions\n"
        "Summarize the key takeaways and actionable insights from the analysis."
    )
    return prompt

def save_readme(insights):
    """
    Saves the generated README.md content to a file.
    Args:
        insights (str): The content to save.
    """
    with open("README.md", "w") as f:
        f.write(insights)
    print("Saved README.md")

def main():
    """
    Main function to orchestrate the analysis, visualization, and README.md generation.
    """
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
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
