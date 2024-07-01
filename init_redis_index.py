import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Set up Azure OpenAI Embeddings
embedding = OpenAIEmbeddings(
    deployment=os.environ.get("DEPLOYMENT_NAME"),
    model=os.environ.get("MODEL_NAME"),
    openai_api_base=os.environ.get("OPENAI_ENDPOINT"),
    openai_api_type="azure",
    openai_api_key=os.environ.get("OPENAI_KEY"),
    openai_api_version="2023-05-15",
    chunk_size=16
)

# In-memory data store
data_store = {}

# Function to add data to the store
def add_to_store(key, value):
    data_store[key] = value

# Example data
example_data = [
    {"Title": "Movie 1", "Description": "Description 1", "Genre": "Action", "Year": 2020},
    {"Title": "Movie 2", "Description": "Description 2", "Genre": "Drama", "Year": 2019},
    # Add more data as needed
]

# Add example data to the store
for item in example_data:
    key = item["Title"]
    value = item
    add_to_store(key, value)

print("In-memory data store initialized successfully.")
