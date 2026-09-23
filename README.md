# Zepto Data & AI Platform — Capstone Project

**Student / Author**: Manikanta Reddy Reddygari  
**GitHub**: [@manikantareddyreddygari-stack](https://github.com/manikantareddyreddygari-stack)  
**Program**: Certificate Program in Artificial Intelligence & Machine Learning  

---

## Project Overview

For this capstone project, I developed an end-to-end data and machine learning platform simulating core analytics and AI workflows at Zepto. Instead of treating these as three isolated homework tasks, I connected them into one single repository:

1. **/data_pipeline**: Scrapes raw book catalog data from a public practice site, cleans and normalizes it, enriches prices with a fixed currency exchange rate, stores it in a relational SQLite database with foreign keys, and validates SQL queries against in-memory Pandas joins.
2. **/analytics**: Profiles the classic Titanic passenger dataset, handles missing data using explicit percentage-based thresholds, presents a multi-chart visual data story, trains three baseline classifiers using leakage-free Scikit-Learn pipelines, evaluates class imbalance strategies (SMOTE vs. class weights), tunes Random Forest hyperparameters with out-of-bag scoring, and explores ticket fare prediction through multivariate linear regression.
3. **/support_assistant**: Implements a grounded GenAI customer support service for Zepto's delivery and store policies using LangGraph for stateful intent routing, ChromaDB with local sentence embeddings (`all-MiniLM-L6-v2`) for retrieval, deterministic offline mock execution (`MOCK_LLM=1`), and a containerized FastAPI endpoint.

---

## Repository Structure

```text
zepto-data-ai-platform/
├── .gitignore
├── requirements.txt                   # Consolidated root dependencies
├── README.md                          # Main project documentation & design rationale
├── data_pipeline/
│   ├── README.md                      # Module 1 guide, schema & query explanations
│   ├── scraper.py                     # BeautifulSoup scraping logic
│   ├── db_loader.py                   # SQLite schema creation & query functions
│   ├── queries.sql                    # Standalone SQL analytics queries
│   ├── run_pipeline.py                # Single script to run Module 1 start to finish
│   ├── data_pipeline.ipynb            # Jupyter notebook walkthrough
│   └── zepto_catalog.db               # Generated normalized SQLite database
├── analytics/
│   ├── README.md                      # Module 2 write-up, metric tables & interpretations
│   ├── titanic.csv                    # Committed offline fallback dataset
│   ├── eda_pipeline.py                # Data profiling, threshold cleaning & EDA script
│   ├── modeling_pipeline.py           # Classifier training, SMOTE, tuning & regression script
│   ├── test_inference.py              # Verification script testing raw data inference
│   ├── 01_eda.ipynb                   # Part A interactive EDA notebook
│   ├── 02_modeling.ipynb              # Part B interactive modeling notebook
│   ├── best_titanic_pipeline.joblib   # Persisted ColumnTransformer + model pipeline
│   └── charts/                        # Saved PNG plots for quick reference
└── support_assistant/
    ├── README.md                      # Module 3 RAG architecture walkthrough & transcripts
    ├── Dockerfile                     # Docker container configuration
    ├── config.py                      # Prompt templates & MOCK_LLM toggle
    ├── schemas.py                     # Pydantic QueryRequest & QueryResponse models
    ├── vector_store.py                # Local ChromaDB indexing & MiniLM embeddings
    ├── graph.py                       # LangGraph StateGraph nodes & conditional edge router
    ├── main.py                        # FastAPI application with POST /ask and GET /health
    ├── test_assistant.py              # Automated test runner with raw JSON transcripts
    └── docs/                          # 8 Zepto policy files (doc_01.txt to doc_08.txt)
```

---

## Setup & How to Run

I chose to maintain **one consolidated `requirements.txt`** at the repository root so you can set up a single virtual environment and run any of the three modules without dependency conflicts.

### 1. Initial Setup
```bash
# Clone the repository
git clone https://github.com/manikantareddyreddygari-stack/zepto-data-ai-platform..git
cd zepto-data-ai-platform

# Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

### 2. Running Module 1: Data Pipeline
To scrape the catalog, clean the records, build `zepto_catalog.db`, execute the 5+ SQL queries, and verify the Pandas merge equivalence:
```bash
python data_pipeline/run_pipeline.py
```
*You can also open and run through `data_pipeline/data_pipeline.ipynb`.*

---

### 3. Running Module 2: Analytics Pipeline
To run the exploratory data analysis, generate charts, train the models, and test the saved pipeline:
```bash
# Part A: Run profiling, threshold cleaning, and generate EDA charts
python analytics/eda_pipeline.py

# Part B: Run classifier training, SMOTE comparison, tuning, regression, and save pipeline
python analytics/modeling_pipeline.py

# Verification: Test reloading the saved joblib pipeline on raw input data
python analytics/test_inference.py
```
*Alternatively, run `analytics/01_eda.ipynb` followed by `analytics/02_modeling.ipynb`.*

---

### 4. Running Module 3: Support Assistant
The assistant defaults to `MOCK_LLM=1`, which runs 100% locally and offline without needing any API key or paid account.

```bash
# Run the automated endpoint tests (tests both policy & general queries)
python support_assistant/test_assistant.py

# Start the live FastAPI server on port 7860
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```
- Interactive Swagger docs: [http://localhost:7860/docs](http://localhost:7860/docs)
- Health check: [http://localhost:7860/health](http://localhost:7860/health)

#### Running with Docker:
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant
docker run -p 7860:7860 zepto-support-assistant
```

---

## My Key Decisions & Engineering Notes

### 1. Data Pipeline Design Decisions
- **Fixed Currency Baseline**: Per project requirements, I used the fixed artificial exchange rate **$1\text{ GBP} = 105.50\text{ INR}$** directly in code so that results are consistent and don't depend on internet access at grading time. I also included an optional helper function that tests an HTTP status code check against a public rate API with a safe fallback to $105.50$.
- **Database Normalization**: I created two tables—`categories` and `books`—linked by `category_id`. Rather than storing redundant category strings on every book row, separating them follows standard third-normal-form relational design and prevents update anomalies.
- **Data Cleaning Strategy**: I stripped currency symbols with regular expressions and mapped word ratings (`"One"`–`"Five"`) to integers $1$–$5$. If a row had missing or corrupted price text, I used median imputation to prevent single missing values from wiping out catalog rows.
- **SQL vs. Pandas Equivalence**: I tested whether querying a relational join through SQL (`pd.read_sql`) produced identical rows and values compared to loading the raw tables and running `pd.merge()` in memory. Using `df_sql.equals(df_merged)`, I confirmed both return identical DataFrames (`True`).

### 2. Analytics Pipeline Design Decisions
- **Offline Dataset Fallback**: The Titanic dataset is fetched once using `sns.load_dataset('titanic')` and immediately saved to `titanic.csv`. Both my EDA script and modeling scripts read from `titanic.csv`, ensuring the grader can run everything without an internet connection.
- **Missing Value Handling Rules**:
  - `embarked` / `embark_town` ($0.22\%$ missing): Under the $5\%$ threshold, so dropping the $2$ affected rows is the cleanest choice without losing statistical power.
  - `age` ($19.87\%$ missing): Within the $5\%-30\%$ threshold, so I imputed using the training median ($28.00$ years) because median is robust to extreme outliers.
  - `deck` ($77.22\%$ missing): Exceeds the $30\%$ threshold by a wide margin. Imputing over three-quarters of missing values would introduce substantial noise, so I decided to drop the column entirely.
- **Fare Skewness**: By comparing $\text{Mean } (32.10) > \text{Median } (14.45) > \text{Mode } (8.05)$, I confirmed that the fare distribution is heavily right-skewed due to luxury ticket outliers.
- **Correlation Filtering**: I restricted the correlation heatmap to the 6 strictly numeric measured columns (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`), intentionally omitting `adult_male` and `alone` since they are derived boolean combinations of sex/age and family size.
- **Leakage Prevention**: All preprocessing (imputation, one-hot encoding, and scaling) is wrapped inside a Scikit-Learn `ColumnTransformer` and fit strictly on `X_train`. The test fold is only transformed.
- **Model Choice**: Random Forest was my top recommendation for deployment. While Logistic Regression had slightly higher linear accuracy, Random Forest achieved an Out-of-Bag (OOB) score of $0.8073$, handled non-linear interactions naturally, and provided balanced precision ($0.7619$) and recall ($0.7059$).
- **Regression Heteroscedasticity**: When predicting `fare` with multivariate linear regression, the residual plot showed a classic fan/funnel shape (residuals widen as predicted fares increase), proving heteroscedasticity in ticket pricing.

### 3. Support Assistant Design Decisions
- **Deterministic Mock Baseline**: By default, `MOCK_LLM=1` runs the whole service without any paid or cloud LLM accounts. Keyword matching determines intent, real cosine similarity runs on local ChromaDB embeddings, and answers are returned using the required template format: `Based on the retrieved context: <snippet>`.
- **Local Embeddings**: Used `sentence-transformers/all-MiniLM-L6-v2` which downloads lightweight embeddings locally and stores them in ChromaDB on disk, keeping the entire service self-contained.
- **LangGraph StateGraph Routing**: Structured the assistant into 3 distinct graph nodes:
  - `classify_intent`: Checks for keywords (`delivery`, `return`, `refund`, `membership`, etc.) to distinguish policy questions from general chat.
  - `retrieve_and_answer`: Pulls the top-3 matching chunks from ChromaDB and formats the grounded answer.
  - `direct_answer`: Returns a friendly canned refusal message for non-policy questions.
- **Structured Output**: Used a Pydantic `QueryResponse` model (`answer: str`, `sources: list[str]`, `confidence: float`). In real-LLM mode (`MOCK_LLM=0`), a retry loop attempts up to 2 corrective retries if the LLM output fails schema validation.

---

## Git Workflow

Development was organized using a feature-branch workflow. The data engineering module was built on `feature/data-pipeline` across multiple atomic commits, and then merged back into `main` via a non-fast-forward merge commit.

You can view the full branch and merge history on GitHub or by running:
```bash
git log --graph --all
```
