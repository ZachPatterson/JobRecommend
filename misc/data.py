import pandas as pd
import os
import re
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
import time

# Load environment variables from .env file
load_dotenv()

# Set up OpenCage Geocoder
opencage_key = os.getenv("OPENCAGE_KEY")  # Ensure your .env file contains OPENCAGE_KEY
geocoder = OpenCageGeocode(opencage_key)

# Function to normalize text
def normalize_text(s):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

# Read in the data
df = pd.read_csv(os.path.join(os.getcwd(), 'data/Client Knowledge Base.csv'), encoding='ISO-8859-1')

# Clean and normalize
df.insert(0, 'id', range(0, len(df)))

# Normalize addresses
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

df['City'] = df['Physical Address'].apply(lambda x: extract_city(x) if pd.notna(x) else "")
df['Zip'] = df['Physical Address'].apply(lambda x: extract_zip(x) if pd.notna(x) else "")

# Clean categorical columns
def clean_categorical_column(column):
    return column.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('&', 'and').str.replace('/', '_').str.replace('\\', '_')

df['Status'] = clean_categorical_column(df['Status'])
df['Vertical'] = clean_categorical_column(df['Vertical'])
df['Sub Vertical'] = clean_categorical_column(df['Sub Vertical'])
df['Functional Expertise'] = clean_categorical_column(df['Functional Expertise'])
df['Type'] = clean_categorical_column(df['Type'])

# Create a description column by concatenating relevant columns
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

# Apply normalization
df['Description'] = df['Description'].astype(str).apply(lambda x: normalize_text(x))

# Create info column with descriptive descriptions about the type of person that would be a good fit for these roles
df['Info'] = df.apply(lambda row: (
    f"A suitable candidate for {row['Position']} at {row['Client Name']} would likely have experience in {row['Vertical']} "
    f"and {row['Sub Vertical']} verticals, with at least {row['Min High Pressure Phone Sales Experience  (Months)']} months of "
    f"high pressure phone sales experience and {row['Min Experience in selling In specified vertical (Months)']} months of experience in "
    f"selling in the specified vertical. The role requires working in {row['Type']} setting with office hours from {row['Office Hours']}. "
    f"Additional skills or certifications such as FFM, License, and Health Sherpa are preferred. "
    f"The position is available in the following states: {row['States']}."
), axis=1)

# Function to get latitude and longitude
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

# Get latitudes and longitudes for job locations
df['LatLon'] = df['Physical Address'].apply(lambda x: get_lat_lon(x) if pd.notna(x) and x.strip() else (None, None))
df[['Latitude', 'Longitude']] = pd.DataFrame(df['LatLon'].tolist(), index=df.index)

# Save the processed data to a new CSV file
df.to_csv(os.path.join(os.getcwd(), 'data/Client Knowledge Base Processed.csv'), index=False)
print("Data processing complete.")
