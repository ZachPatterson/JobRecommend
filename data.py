import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
import tiktoken
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore

def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    # remove all instances of multiple spaces
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

# Read in the data
df = pd.read_csv(os.path.join(os.getcwd(), 'data\Client_Descriptions_Separate_Columns.csv'))

# Clean and normalize
df = df[df.Status == 'Open']
df.insert(0, 'id', range(0, len(df)))

pd.options.mode.chained_assignment = None

# Apply normalization
df['Description'] = df['Description'].astype(str).apply(lambda x: normalize_text(x))

# Check how many tokens it will require to encode all the descriptions
tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df["Description"].apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens < 8192]

print('Number of jobs: ' + str(len(df)))  # print number of jobs remaining in dataset
print('Number of tokens required:' + str(df['n_tokens'].sum()))  # print number of tokens


# load everything into Langchain
from langchain.document_loaders import DataFrameLoader
loader = DataFrameLoader(df, page_content_column="Plot" )
position_list = loader.load()

# # create embeddings and load redis

# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores.redis import Redis as RedisVectorStore


# # we will use OpenAI as our embeddings provider
# embedding = OpenAIEmbeddings(
#     deployment="<your-deployment-name>", # your name for the model deployment (e.g. "my-ml-model")
#     model="<your-deployment-model>", # name of the specific ML model (e.g. "text-embedding-ada-002" )
#     openai_api_base="<your-azureopenai-endpoint>",
#     openai_api_type="azure", # you need this if you're using Azure Open AI service
#     openai_api_key="<your-azureopenai-key>",
#     openai_api_version="2023-05-15",
#     chunk_size=16 # current limit with Azure OpenAI service. This will likely increase in the future.
#     )

# # name of the Redis search index to create
# index_name = "positionindex"

# REDIS_ENDPOINT = "<your-endpoint>" # must include port at the end. e.g. "redisdemo.eastus.redisenterprise.cache.azure.net:10000"
# REDIS_PASSWORD = "<your-password>"

# # create a connection string for the Redis Vector Store. Uses Redis-py format: https://redis-py.readthedocs.io/en/stable/connections.html#redis.Redis.from_url
# # This example assumes TLS is enabled. If not, use "redis://" instead of "rediss://
# redis_url = "rediss://:" + REDIS_PASSWORD + "@"+ REDIS_ENDPOINT

# # create and load redis with documents
# vectorstore = RedisVectorStore.from_documents(
#     documents=position_list,
#     embedding=embedding,
#     index_name=index_name,
#     redis_url=redis_url
# )

# # This may take up to 10 minutes to complete. 

# vectorstore.write_schema("redis_schema.yaml")
# # this saves a copy of the vectorstore schema so you can load it later without having to re-create the vectorstore


# # Basic similarity query
# results = vectorstore.similarity_search_with_score("dogs playing basketball")
# print('Title: '+ str(results[0][0].metadata['Title']))
# print('Vector Distance: ' + str(results[0][1]))

# # Hybrid similarity query with a genre filter first
# from langchain.vectorstores.redis import RedisText, RedisNum, RedisTag

# genre_filter = RedisText("Genre") == "family"
# results = vectorstore.similarity_search_with_score("dogs playing basketball", filter=genre_filter)
# topanswer = results
# print('Title: '+ str(results[0][0].metadata['Title']))
# print('Vector Distance: ' + str(results[0][1]))
