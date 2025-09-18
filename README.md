# Influencer Insights - YouTube Comment Analyzer
#### 📋 Project Overview
Influencer Insights is a Chrome extension designed to help content creators analyze and understand audience sentiment from YouTube video comments. This tool addresses the challenge influencers face in manually processing large volumes of comments by providing automated sentiment analysis, comment summarization, and valuable insights.  
 
#### 🎯 Business Context
We are "Influence Boost Inc.," an influencer management company seeking to expand our network by attracting more influencers to our platform. With a limited marketing budget, we're offering this solution to address a significant pain point for influencers - managing and interpreting vast amounts of comment feedback.  
  
#### ✨ Key Features
1. Sentiment Analysis of Comments  
   - Real-time sentiment classification (**positive, neutral, negative**)  
   - Sentiment distribution visualization with intuitive charts  
   - Detailed sentiment insights with comment drill-down  

2. Additional Analysis Features  
   - **Word cloud visualization** of frequently used words and phrases  
   - **Average comment length** calculation to gauge engagement depth  
   - **Data export functionality** (PDF, CSV) for further analysis  
  
#### 🛠️ Technology Stack
**Backend & Data Processing**  
- **Python** - Core programming language  
- **Flask** - RESTful API development  
- **scikit-learn** - Machine learning algorithms  
- **NLTK & spaCy** - Natural language processing  
- **Pandas & NumPy** - Data manipulation and analysis  

**Frontend & Extension**  
- **JavaScript** - Chrome extension development  
- **HTML/CSS** - Interface structure and styling  
- **D3.js** - Data visualizations and word clouds  

**DevOps & MLOps**  
- **Docker** - Containerization and deployment  
- **DVC** - Data version control  
- **MLflow** - Experiment tracking and model registry  
- **AWS** - Cloud infrastructure (**EC2, ECR, S3, CodeDeploy**)  
- **GitHub Actions** - CI/CD pipeline automation  
 
#### 📊 Workflow
- **Data Collection** - Gather YouTube comments through the Chrome extension  
- **Data Preprocessing** - Clean and prepare comment data for analysis  
- **EDA** - Exploratory data analysis to understand data patterns  
- **Model Building** - Develop and tune sentiment analysis models  
- **DVC Pipeline** - Create reproducible data pipelines  
- **Model Registration** - Register best-performing models in MLflow  
- **API Development** - Build Flask/FastAPI endpoints for the extension  
- **Chrome Plugin Development** - Create the user interface and functionality  
- **CI/CD Pipeline** - Set up automated testing and deployment  
- **Testing** - Comprehensive testing of all components  
- **Dockerization** - Containerize the application  
- **AWS Deployment** - Deploy to cloud infrastructure  
 
#### 🚀 Challenges Addressed
- Multi-language comment processing  
- Handling slang, emojis, and informal language  
- Detecting sarcastic comments  
- Managing evolving language usage (concept drift)  
- Ensuring privacy and data compliance  
- Dealing with spam and bot-generated comments  
- Building efficient models with noisy, imbalanced data  
- Maintaining low latency for real-time analysis  

#### 📁 Project Structure

    yt_commit_analysis/
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── lgbm_model.pkl
    │   ├── vectorizer.pkl
    │   └── experiment_info.json
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── flask_app          <- Flask application for serving the model
    ├── scripts            <- Utility scripts
    │   ├── promote_model.py
    │   ├── test_flask_api.py
    │   ├── test_load_model.py
    │   ├── test_model_performance.py
    │   └── test_model_signature.py
    ├── dvc.yaml           <- DVC pipeline configuration
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    └── .github/workflows  <- GitHub Actions workflows



 
#### 🏗️ Installation & Setup
**Prerequisites**  
- **Python 3.11+**  
- **java script** (for extension development)  
- **Chrome browser**  
- **AWS account** (for deployment)  

**Local Development**  
- Clone the repository  
- Install dependencies: `pip install -r requirements.txt`  
- Set up environment variables  
- Run dev server: `python src/backend/app.py`  
- Load the extension in Chrome developer mode  

**Production Deployment**  
- Build Docker images: `docker build -t influencer-insights .`  
- Push to container registry  
- Deploy to AWS using CodeDeploy  
- Configure auto-scaling groups for load management  
 
#### 📈 Model Performance
Our sentiment analysis model has been trained on **3.7K labeled comments**, achieving strong performance in classifying **positive, neutral, and negative** sentiments across diverse comment styles and languages.  

---
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
