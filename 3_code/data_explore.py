"""
Data Exploration Script for arXiv Dataset
Project: Smart Research Assistant Chatbot (Data601-25A)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Configuration
BACKUP_METADATA_PATH = Path("../6_testing_data/metadata/metadata.csv") 
OUTPUT_PLOT_DIR = Path("../6_testing_data/exploration_results")  # Directory to save plots
OUTPUT_REPORT_PATH = Path("../6_testing_data/dataset_exploration_report.md")  # Path to save analysis report

# Create output directory if it doesn't exist
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 1: Load and Validate Data
def load_and_validate_data(metadata_path):
    """
    Load metadata CSV and perform basic validation (check for missing values, data types)
    """
    try:
        df = pd.read_csv(metadata_path, encoding="utf-8")
        print(f"Successfully loaded metadata: {len(df)} papers")
    except Exception as e:
        raise ValueError(f"Failed to load metadata: {str(e)}") from e

    # Basic validation checks
    validation_results = {
        "total_papers": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_ids": df["id"].duplicated().sum(),
        "data_types": df.dtypes.to_dict()
    }

    # Print validation summary
    print("Data Validation Summary:")
    for key, value in validation_results.items():
        print(f"  - {key}: {value}")

    # Drop duplicates (if any)
    if validation_results["duplicate_ids"] > 0:
        df = df.drop_duplicates(subset="id", keep="first")
        print(f"Removed {validation_results['duplicate_ids']} duplicate papers. Remaining: {len(df)}")

    return df, validation_results

# 2: Exploratory Data Analysis 
def perform_eda(df):
    """
    Perform core EDA: distribution of categories, submission time, abstract length, etc.
    """
    # Convert submit_date to datetime for time-based analysis
    df["submit_date"] = pd.to_datetime(df["submit_date"], errors="coerce")
    df["submit_year"] = df["submit_date"].dt.year
    df["submit_month"] = df["submit_date"].dt.month

    # 1. Category distribution 
    all_categories = []
    for cats in df["categories"].dropna():
        all_categories.extend([cat.strip() for cat in str(cats).split(",")])
    category_counts = pd.Series(all_categories).value_counts().head(10)  # Top 10 categories

    # 2. Submission year distribution
    year_counts = df["submit_year"].value_counts().sort_index()

    # 3. Abstract length analysis (number of characters)
    df["abstract_length"] = df["abstract"].fillna("").apply(len)
    abstract_stats = {
        "mean_length": df["abstract_length"].mean(),
        "median_length": df["abstract_length"].median(),
        "min_length": df["abstract_length"].min(),
        "max_length": df["abstract_length"].max()
    }

    # 4. Author count distribution (approximate: split by comma)
    df["author_count"] = df["authors"].fillna("").apply(lambda x: len(str(x).split(",")))
    author_count_stats = df["author_count"].value_counts().head(8)

    # Compile EDA results
    eda_results = {
        "category_counts": category_counts,
        "year_counts": year_counts,
        "abstract_stats": abstract_stats,
        "author_count_stats": author_count_stats,
        "time_range": {
            "earliest_submit": df["submit_date"].min(),
            "latest_submit": df["submit_date"].max()
        }
    }

    return eda_results, df

# 3: Visualization
def generate_visualizations(df, eda_results, output_dir):
    """
    Generate and save key visualizations (bar charts, histograms, line charts)
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10

    # Create a 2x2 subplot for core visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("arXiv Dataset Exploration (Sample: 12,000 Papers)", fontsize=14, fontweight="bold")

    # 1. Top 10 Categories (Bar Chart)
    eda_results["category_counts"].plot(
        kind="bar", ax=axes[0, 0], color="#1f77b4"
    )
    axes[0, 0].set_title("Top 10 Paper Categories")
    axes[0, 0].set_xlabel("Category")
    axes[0, 0].set_ylabel("Number of Papers")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Submission Year Distribution (Line Chart)
    eda_results["year_counts"].plot(
        kind="line", ax=axes[0, 1], marker="o", linewidth=2, color="#ff7f0e"
    )
    axes[0, 1].set_title("Papers by Submission Year")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Number of Papers")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Abstract Length Distribution (Histogram)
    df["abstract_length"].clip(upper=3000).plot(  # Clip outliers for better visualization
        kind="hist", ax=axes[1, 0], bins=50, color="#2ca02c", alpha=0.7
    )
    axes[1, 0].set_title("Abstract Length Distribution (Max 3000 Chars)")
    axes[1, 0].set_xlabel("Abstract Length (Characters)")
    axes[1, 0].set_ylabel("Frequency")

    # 4. Author Count Distribution (Bar Chart)
    eda_results["author_count_stats"].plot(
        kind="bar", ax=axes[1, 1], color="#d62728"
    )
    axes[1, 1].set_title("Top 8 Author Count Distribution")
    axes[1, 1].set_xlabel("Number of Authors")
    axes[1, 1].set_ylabel("Number of Papers")
    axes[1, 1].tick_params(axis="x", rotation=0)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_exploration_plots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved to: {output_dir / 'dataset_exploration_plots.png'}")

# 4: Generate Analysis Report
def generate_report(validation_results, eda_results, output_path):
    report_content = f"""# Dataset Exploration Report
## Smart Research Assistant Chatbot Project 
Sample Size: {validation_results['total_papers']} papers (backup from full dataset)

---

## 1. Data Validation Results
| Metric | Value |
|--------|-------|
| Total Papers | {validation_results['total_papers']} |
| Columns | {', '.join(validation_results['columns'])} |
| Duplicate IDs | {validation_results['duplicate_ids']} |
| Missing Values | {validation_results['missing_values']} |

---

## 2. Key EDA Findings
### 2.1 Category Distribution
Top 5 categories:
{chr(10).join([f"- {cat}: {count} papers" for cat, count in eda_results['category_counts'].head(5).items()])}

### 2.2 Time Range
- Earliest Submission: {eda_results['time_range']['earliest_submit'].strftime('%Y-%m-%d') if pd.notna(eda_results['time_range']['earliest_submit']) else 'N/A'}
- Latest Submission: {eda_results['time_range']['latest_submit'].strftime('%Y-%m-%d') if pd.notna(eda_results['time_range']['latest_submit']) else 'N/A'}
- Most Active Year: {eda_results['year_counts'].idxmax()} ({eda_results['year_counts'].max()} papers)

### 2.3 Abstract Length Statistics
| Statistic | Value |
|-----------|-------|
| Mean Length | {eda_results['abstract_stats']['mean_length']:.0f} chars |
| Median Length | {eda_results['abstract_stats']['median_length']:.0f} chars |
| Min Length | {eda_results['abstract_stats']['min_length']:.0f} chars |
| Max Length | {eda_results['abstract_stats']['max_length']:.0f} chars |

### 2.4 Author Distribution
Most common author count: {eda_results['author_count_stats'].idxmax()} authors ({eda_results['author_count_stats'].max()} papers)

---

## 3. Preprocessing Pipeline Recommendations
Based on exploration, the following preprocessing steps are recommended:

### 3.1 Text Extraction
- Use `pdfplumber` to extract full text from PDFs (more reliable than PyPDF2 for academic papers)
- Handle edge cases: corrupted PDFs, scanned documents (flag for manual review)

### 3.2 Data Cleaning
- Remove papers with empty abstracts/ titles (missing values identified in validation)
- Strip special characters, extra whitespace, and non-ASCII characters (if needed for embedding quality)
- Standardize category names (e.g., lowercase, remove redundant tags)

### 3.3 Text Chunking
- Chunk full text into segments of 500-1000 characters (balances context retention and embedding efficiency)
- Bind chunk-level metadata (paper ID, category, submit year) for retrieval filtering

### 3.4 Embedding Generation
- Use `Sentence-Transformers/all-MiniLM-L6-v2`
- Generate embeddings for both abstracts and full-text chunks (for hybrid retrieval)

---

## 4. Notes for Full Dataset
- The full dataset target is 100,000+ papers (Week 1 Task). This sample is representative of the full dataset structure.
- Monitor category balance in the full dataset (ensure coverage of cs.AI, cs.LG, cs.CV, etc.)
- Increase validation checks for the full dataset (e.g., PDF file existence, text extraction success rate)
"""

    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Analysis report saved to: {output_path}")

# Main Workflow
if __name__ == "__main__":
    print("Starting arXiv Dataset Exploration (Week 1 Task)")

    # Step 1: Load and validate data
    df, validation_results = load_and_validate_data(BACKUP_METADATA_PATH)

    # Step 2: Perform EDA
    eda_results, df_processed = perform_eda(df)

    # Step 3: Generate visualizations
    generate_visualizations(df_processed, eda_results, OUTPUT_PLOT_DIR)

    # Step 4: Generate analysis report
    generate_report(validation_results, eda_results, OUTPUT_REPORT_PATH)
    print("Outputs:")
    print(f"  - Plots: {OUTPUT_PLOT_DIR}")
    print(f"  - Report: {OUTPUT_REPORT_PATH}")
