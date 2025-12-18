import os
import numpy as np
import pandas as pd
import warnings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import warnings
warnings.filterwarnings('ignore')

chat_client = ChatOpenAI(
    model="gpt-4o-mini",
    api_key = OPENAI_API_KEY,
    temperature=0
)

df = pd.read_csv("Data/provider.csv")
df.head()

metadata = {
    "provider": "The dataset contains provider-level billing and reimbursement details for specific services or procedures, capturing service frequency, billed amounts, and payments for comparative analysis across providers and service types.",
    "claims": "The dataset contains historical insurance claim records with policyholder demographics, claim history, and related features designed for predictive modeling of future claim likelihood.",
    "marketing": "The dataset contains information on 2,206 customers of XYZ company, including customer profiles, product preferences, marketing campaign performance, and sales channel effectiveness."
}


data_dir = "Data"
datasets = {}

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        name = file.replace(".csv", "")
        datasets[name] = pd.read_csv(os.path.join(data_dir, file))


dataset_selector_prompt = PromptTemplate(
    input_variables=["query", "metadata"],
    template=(
        "You are a data reasoning assistant. You have access to the following datasets:\n\n"
        "{metadata}\n\n"
        "Given the user query:\n'{query}'\n\n"
        "Which datasets are needed to answer this? "
        "Return a comma-separated list of dataset names only (no explanations)."
    )
)

def select_datasets(chat_client, query, metadata):
    metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    formatted_prompt = dataset_selector_prompt.format(query=query, metadata=metadata_text)
    response = chat_client.invoke(formatted_prompt)
    selected = [ds.strip() for ds in response.cjontent.split(",") if ds.strip() in metadata.keys()]
    return selected


query = "Which provider is charging more over the past 3 months?"

selected = select_datasets(chat_client, query, metadata)
print("Datasets selected:", selected)

dfs = [datasets[name] for name in selected]
if len(dfs) > 1:
    combined_df = pd.concat(dfs, axis=1, join="inner")
else:
    combined_df = dfs[0]