# Zepto Data & AI Platform

**Capstone Project — Certificate Program in Artificial Intelligence and Machine Learning**  
*Author*: AI/ML Engineering Guild  
*Repository Structure*: Single unified repository comprising three interconnected capabilities:
- `/data_pipeline`: Automated catalog scraping, data cleaning, fixed-rate currency conversion, normalized relational SQLite warehouse ingestion, analytical SQL queries, and mathematical equivalence verification.
- `/analytics`: End-to-end dataset profiling, threshold-based imputation/cleaning, univariate/bivariate analysis, multivariate data storytelling, stratified predictive modeling (Logistic Regression, Decision Tree, Random Forest), class imbalance mitigation (SMOTE, class weights), hyperparameter tuning, regression side-task, and pipeline persistence.
- `/support_assistant`: Grounded GenAI customer support service leveraging LangGraph intent routing, local ChromaDB cosine similarity retrieval (using `all-MiniLM-L6-v2`), deterministic offline mock mode (`MOCK_LLM=1`), strict Pydantic JSON schema enforcement, and containerized FastAPI serving.

---

## Repository Structure

```
zepto-data-ai-platform/
├── .gitignore
├── requirements.txt                   # Consolidated root requirements for all modules
├── README.md                          # Root system documentation & design decisions
├── data_pipeline/
│   ├── README.md                      # Module 1 guide & schema specifications
│   ├── scraper.py                     # Catalog web scraper (requests + BeautifulSoup)
│   ├── db_loader.py                   # Normalized SQLite loader & SQL/Pandas engine
│   ├── queries.sql                    # Executed SQL queries
│   ├── run_pipeline.py                # End-to-end execution script
│   ├── data_pipeline.ipynb            # Interactive execution notebook
│   └── zepto_catalog.db               # Normalized SQLite relational database
├── analytics/
│   ├── README.md                      # Module 2 documentation, metrics & interpretations
│   ├── titanic.csv                    # Committed offline fallback dataset
│   ├── eda_pipeline.py                # Part A: Profiling, cleaning, IQR, bivariate, multivariate
│   ├── modeling_pipeline.py           # Part B: Preprocessing, 3 classifiers, tuning, regression
│   ├── test_inference.py              # Standalone joblib raw inference verification
│   ├── 01_eda.ipynb                   # Interactive EDA notebook
│   ├── 02_modeling.ipynb              # Interactive Predictive Modeling notebook
│   ├── best_titanic_pipeline.joblib   # Persisted full fitted pipeline
│   └── charts/                        # Generated high-resolution visualization artifacts
└── support_assistant/
    ├── README.md                      # Module 3 RAG architecture & API documentation
    ├── Dockerfile                     # Containerization specification for FastAPI
    ├── config.py                      # Configurations, MOCK_LLM toggle, structured prompts
    ├── schemas.py                     # Pydantic QueryRequest & QueryResponse schemas
    ├── vector_store.py                # Local all-MiniLM-L6-v2 embedding & ChromaDB engine
    ├── graph.py                       # LangGraph StateGraph (classify, retrieve, direct)
    ├── main.py                        # FastAPI application serving POST /ask & GET /health
    ├── test_assistant.py              # End-to-end endpoint verification & test suite
    └── docs/
        ├── doc_01.txt ... doc_08.txt  # Exact 8 Zepto policy files
```

---

## Setup & Installation

This project utilizes a **single consolidated `requirements.txt`** located at the repository root to ensure unified dependency management across all three capabilities.

### 1. Environment Initialization
```bash
# Clone the repository
git clone <repository_url>
cd zepto-data-ai-platform

# Create and activate virtual environment (Python 3.10+ recommended)
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install all project dependencies
pip install -r requirements.txt
```

---

## How to Run Each Module End to End

### Module 1: Data Pipeline (`/data_pipeline`)
```bash
# Execute end-to-end scraping, SQLite loading, and query verification
python data_pipeline/run_pipeline.py
```
*Alternatively, open and run `data_pipeline/data_pipeline.ipynb`.*

### Module 2: Analytics Pipeline (`/analytics`)
```bash
# Run Exploratory Data Analysis & visual generation
python analytics/eda_pipeline.py

# Run Predictive Modeling, tuning, regression, and model persistence
python analytics/modeling_pipeline.py

# Verify raw data inference using persisted joblib pipeline
python analytics/test_inference.py
```
*Alternatively, run `analytics/01_eda.ipynb` followed by `analytics/02_modeling.ipynb`.*

### Module 3: Support Assistant (`/support_assistant`)
```bash
# Run automated test suite against FastAPI endpoints (default MOCK_LLM=1)
python support_assistant/test_assistant.py

# Start local FastAPI server on port 7860
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```
- Interactive Swagger UI: `http://localhost:7860/docs`
- Health Check: `http://localhost:7860/health`

#### Run via Docker:
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant
docker run -p 7860:7860 zepto-support-assistant
```

---

## Architectural & Design Decisions

### Module 1: Data Pipeline
- **Fixed Currency Baseline**: Applied the exact required project constant: **$1\text{ GBP} = 105.50\text{ INR}$**, ensuring deterministic and reproducible offline evaluations.
- **Relational Normalization**: Designed a two-table schema (`categories` and `books`) with `category_id` enforcing foreign key integrity. This eliminates text redundancy across catalog entries and aligns with enterprise data warehouse design.
- **Data Hygiene**: Parsed currency symbols via regex, mapped text ratings to integers ($1$–$5$), converted availability to binary flags, and safeguarded numeric pricing with median imputation.
- **Equivalence Verification**: Demonstrated that relational warehouse SQL joins (`pd.read_sql`) and in-memory application merges (`pd.merge`) produce mathematically identical outputs (`True`).

### Module 2: Analytics Pipeline
- **Single Load with Offline Fallback**: Raw Titanic data is fetched once from Seaborn cache/network and committed to `titanic.csv`. Both EDA and modeling read from this offline fallback, preventing redundant network requests.
- **Defensible Imputation Rules**: Evaluated exact missing value percentages per column:
  - $<5\%$ (`embarked` at $0.22\%$) $\to$ dropped affected rows.
  - $5\%-30\%$ (`age` at $19.87\%$) $\to$ median imputation ($28.00$ years) to protect against skew.
  - $>30\%$ (`deck` at $77.22\%$) $\to$ dropped column due to severe sparsity.
- **Leakage Prevention**: Built a scikit-learn `ColumnTransformer` pipeline fitted strictly on training data and applied in transform-only mode to the test split.
- **Rigorous Evaluation**: Demonstrated that Random Forest achieved top-tier balance across Accuracy ($0.803$), F1 ($0.733$), and ROC-AUC ($0.827$). Imbalance comparison confirmed that training-fold SMOTE and balanced class weights significantly elevate minority recall without test contamination.
- **Heteroscedasticity Analysis**: Multivariate regression on `fare` uncovered a pronounced fan-shaped residual spread, confirming heteroscedasticity driven by extreme luxury ticket price variance.
- **Pipeline Deployment**: The complete fitted preprocessing + classifier bundle is persisted via `joblib.dump` and verified to score raw, uncleaned records seamlessly.

### Module 3: Support Assistant
- **Zero-Cost Offline Baseline (`MOCK_LLM=1`)**: Every LLM generation step branches on the `MOCK_LLM` environment variable. By default, the service executes entirely offline using local sentence embeddings (`all-MiniLM-L6-v2`) and local ChromaDB vector indexing.
- **Graph-Based Intent Routing**: Built a 3-node LangGraph `StateGraph` with a conditional edge routing queries to `retrieve_and_answer` or `direct_answer`.
- **Structured Schema & Guardrails**: Enforced a Pydantic `QueryResponse` (`answer`, `sources`, `confidence`) with explicit prompt engineering featuring role definitions, negative constraints, and few-shot formatting.

---

## Git Workflow & Branch History

As required by the project specifications, development was conducted using a clean Git feature-branch workflow. Feature branch `feature/data-pipeline` was created, committed to twice, and merged back into `main` with a merge commit.

Verifiable via `git log --graph --all --oneline`:
```text
*   2ea6436 merge: integrate data pipeline module from feature/data-pipeline
|\  
| * af24a70 feat(data_pipeline): complete notebook, SQLite database artifact, and module README
| * 436b8ac feat(data_pipeline): add catalog scraper, normalized SQLite schema loader, and query suite
|/  
* eb5881d chore: initial repository skeleton with gitignore and requirements
```

---

## Academic Integrity & License
All code, database schemas, predictive models, prompt templates, and written analyses were developed specifically for the Zepto Data & AI Platform Capstone Project.
