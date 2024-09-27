# Azure Housing Market Analysis

This project leverages Azure Machine Learning services, Python, Prompt Flow, MLFlow, Azure AI Services, and Azure DevOps to analyze the housing market using mortgage data and the Zillow API. Our solution provides valuable insights for real estate professionals, investors, and policymakers.

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Use Cases](#use-cases)
- [Examples](#examples)
- [License](#license)

## Project Overview

Our Azure Housing Market Analysis project combines powerful cloud-based machine learning tools with real-estate data to provide accurate predictions and insights into housing market trends. By utilizing Azure's ecosystem, we've created a scalable, efficient, and highly accurate solution for analyzing housing data.

Key features:
- Data ingestion from various sources, including the Zillow API
- Data preprocessing and feature engineering using Azure Machine Learning
- Model training and evaluation using MLFlow
- Integration of large language models (LLMs) using Prompt Flow for natural language processing tasks
- Automated CI/CD pipeline using Azure DevOps
- Deployment of machine learning models as REST APIs
- Interactive dashboards for visualizing insights

## Architecture

Our solution leverages the following Azure services and tools:

1. Azure Machine Learning: For data preparation, model training, and deployment
2. Azure Databricks: For distributed data processing and feature engineering
3. Azure SQL Database: For storing structured data
4. Azure Blob Storage: For storing raw and processed datasets
5. Azure Key Vault: For securely storing API keys and credentials
6. Azure Functions: For serverless compute and data ingestion
7. Azure Container Registry: For storing and managing Docker containers
8. Azure Kubernetes Service (AKS): For deploying and scaling our machine learning models
9. Azure DevOps: For CI/CD pipelines and project management
10. Azure AI Services: For leveraging pre-built AI models and cognitive services

## Project Structure

```
azure-housing-market-analysis/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── src/
│   ├── data/
│   │   ├── ingest_zillow_data.py
│   │   └── preprocess_data.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   ├── visualization/
│   │   └── create_dashboards.py
│   └── api/
│       └── predict_api.py
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_experimentation.ipynb
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_preprocessing.py
│   └── test_model.py
├── prompts/
│   ├── market_analysis.yaml
│   └── price_prediction.yaml
├── config/
│   ├── model_config.yaml
│   └── azure_config.yaml
├── docs/
│   ├── api_documentation.md
│   └── user_guide.md
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/azure-housing-market-analysis.git
   cd azure-housing-market-analysis
   ```

2. Set up a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Set up Azure resources:
   - Create an Azure Machine Learning workspace
   - Set up Azure Databricks workspace
   - Create Azure SQL Database and Blob Storage accounts
   - Configure Azure Key Vault with necessary secrets

4. Configure Azure DevOps:
   - Create a new project in Azure DevOps
   - Set up service connections to your Azure subscription
   - Import the repository into Azure Repos

5. Update the `config/azure_config.yaml` file with your Azure resource details.

## Usage

1. Data Ingestion:
   ```
   python src/data/ingest_zillow_data.py
   ```

2. Data Preprocessing:
   ```
   python src/data/preprocess_data.py
   ```

3. Feature Engineering:
   ```
   python src/features/feature_engineering.py
   ```

4. Model Training:
   ```
   python src/models/train_model.py
   ```

5. Model Evaluation:
   ```
   python src/models/evaluate_model.py
   ```

6. Deploy the prediction API:
   ```
   az ml model deploy -n housing-prediction-api -m housing-model:1 --ic inference-config.yml --dc deployment-config.yml
   ```

## Use Cases

1. **Home Price Prediction**: Predict future home prices based on historical data and current market trends.

2. **Market Trend Analysis**: Identify emerging trends in the housing market, such as hot neighborhoods or property types.

3. **Investment Opportunity Detection**: Highlight properties or areas that may be undervalued and present good investment opportunities.

4. **Mortgage Risk Assessment**: Evaluate the risk associated with mortgage applications based on various factors.

5. **Natural Language Queries**: Use Prompt Flow to allow users to ask questions about the housing market in natural language and receive insights.

6. **Automated Reporting**: Generate periodic reports on market conditions, price trends, and forecasts.

## Examples

### Home Price Prediction

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

# Authenticate to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

# Define and create compute target
compute_name = "housing-cluster"
compute_config = AmlCompute(
    name=compute_name,
    size="STANDARD_DS3_V2",
    min_instances=0,
    max_instances=4,
)
ml_client.begin_create_or_update(compute_config).result()

# Run the training pipeline
pipeline_job = ml_client.jobs.create_or_update(
    job=load_job("pipelines/train_housing_model.yml", ml_client)
)
pipeline_job.wait_for_completion()

# Get the trained model
model = ml_client.models.get(name="housing-price-model", version=1)

# Make predictions
input_data = {
    "zipcode": "98101",
    "sqft": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1985
}
prediction = ml_client.online_endpoints.invoke(
    endpoint_name="housing-prediction",
    deployment_name="blue",
    request_file="./sample_input.json"
)
print(f"Predicted home price: ${prediction['price']:,.2f}")
```

### Market Trend Analysis using Prompt Flow

```python
from promptflow import PFClient

# Initialize Prompt Flow client
pf_client = PFClient()

# Load and run the flow
flow = pf_client.flows.load("prompts/market_analysis.yaml")
result = pf_client.flows.test(flow, inputs={
    "location": "Seattle, WA",
    "timeframe": "next 6 months"
})

print("Market Analysis:")
print(result['analysis'])
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
