# Dataset Exploration Report
## Smart Research Assistant Chatbot Project 
Sample Size: 13586 papers (backup from full dataset)

---

## 1. Data Validation Results
| Metric | Value |
|--------|-------|
| Total Papers | 13586 |
| Columns | id, title, authors, abstract, categories, submit_date, pdf_path |
| Duplicate IDs | 0 |
| Missing Values | {'id': 0, 'title': 0, 'authors': 0, 'abstract': 0, 'categories': 0, 'submit_date': 0, 'pdf_path': 0} |

---

## 2. Key EDA Findings
### 2.1 Category Distribution
Top 5 categories:
- cs.AI: 6088 papers
- cs.LG: 6036 papers
- cs.CV: 5396 papers
- cs.CL: 1370 papers
- stat.ML: 677 papers

### 2.2 Time Range
- Earliest Submission: 2025-11-04
- Latest Submission: 2025-12-24
- Most Active Year: 2025 (13586 papers)

### 2.3 Abstract Length Statistics
| Statistic | Value |
|-----------|-------|
| Mean Length | 1349 chars |
| Median Length | 1352 chars |
| Min Length | 137 chars |
| Max Length | 2447 chars |

### 2.4 Author Distribution
Most common author count: 3 authors (2215 papers)

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
