import pandas as pd
import os
import re
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
import time
import redis
import json
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Constants
OPENCAGE_KEY = os.getenv("OPENCAGE_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
DATA_FILE = os.path.join(os.getcwd(), 'data', 'Client Knowledge Base.csv')
EXPORT_FILE = os.path.join(os.getcwd(), 'data', 'Processed Client Knowledge Base.csv')

# Set up OpenCage Geocoder
geocoder = OpenCageGeocode(OPENCAGE_KEY)

# Set up Azure OpenAI Embeddings
embedding = OpenAIEmbeddings(
    deployment=DEPLOYMENT_NAME,
    model=MODEL_NAME,
    openai_api_base=OPENAI_ENDPOINT,
    openai_api_type="azure",
    openai_api_key=OPENAI_KEY,
    openai_api_version="2023-05-15",
    chunk_size=16
)

# Set up Redis
r = redis.StrictRedis(host=REDIS_ENDPOINT, port=6380, password=REDIS_PASSWORD, ssl=True)

# Methods
def normalize_text(s):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

def extract_city(address):
    try:
        parts = address.split(',')
        return parts[-3].strip() if len(parts) >= 3 else ""
    except:
        return ""

def extract_zip(address):
    try:
        parts = address.split(',')
        zip_code = parts[-1].strip().split()[-1]
        return zip_code if len(zip_code) == 5 else ""
    except:
        return ""

def clean_categorical_column(column):
    return column.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('&', 'and').str.replace('/', '_').str.replace('\\', '_')

def get_lat_lon(address, retries=3, delay=5):
    for _ in range(retries):
        try:
            result = geocoder.geocode(address)
            if result and len(result):
                location = result[0]['geometry']
                return (location['lat'], location['lng'])
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(delay)
    return (None, None)

# Main script
# Clear the Redis cache
r.flushdb()
print("Flushed Redis Cache.")

# Read in the data
df = pd.read_csv(DATA_FILE, encoding='ISO-8859-1')
print("Data read. Cleaning and normalizing.")

# Clean and normalize
df.insert(0, 'id', range(0, len(df)))

df['City'] = df['Physical Address'].apply(lambda x: extract_city(x) if pd.notna(x) else "")
df['Zip'] = df['Physical Address'].apply(lambda x: extract_zip(x) if pd.notna(x) else "")

df['Status'] = clean_categorical_column(df['Status'])
df['Vertical'] = clean_categorical_column(df['Vertical'])
df['Sub Vertical'] = clean_categorical_column(df['Sub Vertical'])
df['Functional Expertise'] = clean_categorical_column(df['Functional Expertise'])
df['Type'] = clean_categorical_column(df['Type'])

df['Description'] = df.apply(lambda row: (
    f"{row['Position']} at {row['Client Name']}, located at {row['Physical Address']} ({row['Zip']}). "
    f"Status: {row['Status']}. Vertical: {row['Vertical']}, Sub Vertical: {row['Sub Vertical']}. "
    f"Functional Expertise: {row['Functional Expertise']}. Type: {row['Type']}. "
    f"Min High Pressure Phone Sales Experience: {row['Min High Pressure Phone Sales Experience  (Months)']} months, "
    f"Min Experience in selling in specified vertical: {row['Min Experience in selling In specified vertical (Months)']} months. "
    f"FFM: {row['FFM']}, License: {row['License']}, Health Sherpa: {row['Health Sherpa']}. "
    f"TLD Experience: {row['TLD Experience']}, States: {row['States']}, Required states: {row['Required states']}, "
    f"Avg Age of current employees: {row['Avg  Age of current employees (years)']} years, "
    f"Office Hours: {row['Office Hours']}. Weekly Base: {row['Weekly Base']}, Draw: {row['Draw']}, "
    f"Commission: {row['Commission']}, Avg Weekly Income: {row['Avg Weekly Income']}, Revenue: {row['Revenue']}. "
    f"Job URL: {row['Job URL']}, Hot: {row['Hot']}, Hire Rate: {row['Hire Rate']}, 4 week attrition rate: {row['4 week attrition rate']}."
), axis=1)

df['Description'] = df['Description'].astype(str).apply(lambda x: normalize_text(x))

df['Info'] = df.apply(lambda row: (
    f"A suitable candidate for {row['Position']} at {row['Client Name']} would likely have experience in {row['Vertical']} "
    f"and {row['Sub Vertical']} verticals, with at least {row['Min High Pressure Phone Sales Experience  (Months)']} months of "
    f"high pressure phone sales experience and {row['Min Experience in selling In specified vertical (Months)']} months of experience in "
    f"selling in the specified vertical. The role requires working in {row['Type']} setting with office hours from {row['Office Hours']}. "
    f"Additional skills or certifications such as FFM, License, and Health Sherpa are preferred. "
    f"The position is available in the following states: {row['States']}."
), axis=1)

df['Candidate Description'] = df['Candidate Description'].astype(str).apply(lambda x: normalize_text(x))

print("Getting LatLon from physical address.")
df['LatLon'] = df['Physical Address'].apply(lambda x: get_lat_lon(x) if pd.notna(x) and x.strip() else (None, None))
df[['Latitude', 'Longitude']] = pd.DataFrame(df['LatLon'].tolist(), index=df.index)

print("Embedding descriptions.")
embeddings = []
for _, row in df.iterrows():
    job_data = {
        'id': row['id'],
        'company': row['Client Name'],
        'position': row['Position'],
        'description': row['Description'],
        'City': row['City'],
        'Zip': row['Zip'],
        'Status': row['Status'],
        'Vertical': row['Vertical'],
        'Sub Vertical': row['Sub Vertical'],
        'Type': row['Type'],
        'Functional Expertise': row['Functional Expertise'],
        'Physical Address': row['Physical Address'],
        'Latitude': row['Latitude'],
        'Longitude': row['Longitude'],
        'License': row['License'],
        'FFM': row['FFM'],
        'embedding': embedding.embed_query(row['Candidate Description'])
    }
    embeddings.append(job_data)
    r.set(f"job:{row['id']}", json.dumps(job_data))

# Create a DataFrame with the embeddings and export to CSV
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.to_csv(EXPORT_FILE, index=False)
print(f"Data processing complete. Exported data to {EXPORT_FILE}.")
