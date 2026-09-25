# Zepto AI/ML Capston

This project brings together three related parts of the capstone:

1. Data pipeline: scraping, cleaning, normalization, and relational storage. See data_pipeline.
2. Analytics pipeline: Titanic profiling, EDA, model building, and regression. See analytics.
3. Support assistant: a local RAG service for Zepto policy questions. See support_assistant.

The dependencies for all three parts are listed in the root `requirements.txt` file.

# Setup

```bash
cd "c:/Users/servi/capstone project"
pip install -r requirements.txt
```

### Running on another laptop

The project uses relative paths, so it can be copied to a different folder or computer. On a new laptop, install Python 3.11 or newer, open a terminal in the project folder, and run:

```bash
# Windows PowerShell
python -m pip install -r requirements.txt
```

The first run needs internet access for the book website, the Titanic dataset download, and the `all-MiniLM-L6-v2` embedding model. Later support-assistant runs reuse the local model cache and ChromaDB files. The application does not need an API key when `MOCK_LLM` is left unset.

Run the modules from the project root using the commands below. If the computer blocks compiled Python libraries with a security policy, the analytics script may need Python installed from the official Python installer or a virtual environment with permission to load native packages.

# Module 1: Data pipeline

```bash
python data_pipeline/pipeline.py
```

This script collects books from four categories on books.toscrape.com, cleans the fields, converts the prices from GBP to INR using 1 GBP = 105.50 INR, and saves the results in `data_pipeline/zepto_books.db`.

# Module 2: Analytics

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/reload_pipeline_check.py
```

The EDA script saves a local copy of the Titanic data, checks missing values and outliers, and creates the charts. The modeling script cleans the data, trains the classifiers and fare regressor, reports the metrics, and saves the selected pipeline to `analytics/best_model_pipeline.joblib`.

# Module 3: Support assistant

```bash
cd support_assistant
python main.py
```

Then call the app with:

The assistant stores local embeddings in ChromaDB and uses LangGraph to route questions. `MOCK_LLM` is enabled by default, so the example works without an API key or an external LLM call.

# Docker

The root-level Dockerfile packages the support assistant as a small local service.

```bash
docker build -t zepto-capstone .
docker run --rm -p 7860:7860 zepto-capstone
```

# Notes on the implementation

- Data pipeline: four category pages are scraped, invalid rows are dropped, and the cleaned data is stored in two related SQLite tables.
- Analytics: the Titanic data is saved locally and reused by the EDA and modeling scripts. The `alive` column is removed because it gives away the target.
- Support assistant: the documents and embeddings stay local. Mock mode handles the answer generation by default.
