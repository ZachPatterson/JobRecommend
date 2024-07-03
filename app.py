import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
import os
from dotenv import load_dotenv
import re
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import docx2txt
from PyPDF2 import PdfReader
from geopy.distance import geodesic
from opencage.geocoder import OpenCageGeocode

# Load environment variables from .env file
load_dotenv()

# Set up Azure OpenAI Embeddings
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME")
MODEL_NAME = os.environ.get("MODEL_NAME")
OPENCAGE_KEY = os.environ.get("OPENCAGE_KEY")

geocoder = OpenCageGeocode(OPENCAGE_KEY)

embedding = OpenAIEmbeddings(
    deployment=DEPLOYMENT_NAME,
    model=MODEL_NAME,
    openai_api_base=OPENAI_ENDPOINT,
    openai_api_type="azure",
    openai_api_key=OPENAI_KEY,
    openai_api_version="2023-05-15",
    chunk_size=16
)

# In-memory data store
data_store = {}

# Function to add data to the store
def add_to_store(key, value):
    data_store[key] = value

# Load the processed CSV file
csv_file_path = 'data/Client Knowledge Base Processed.csv'
df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

# Add data to the in-memory store with embeddings
for _, row in df.iterrows():
    key = f"{row['Client Name']}_{row['Position']}"
    description = row['Description']
    embedding_vector = embedding.embed_query(description)
    add_to_store(key, {
        'id': row['id'],
        'company': row['Client Name'],
        'position': row['Position'],
        'description': description,
        'embedding': embedding_vector,
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
        'FFM': row['FFM']
    })

# Function to extract text from uploaded files
def extract_text_from_file(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    else:
        return ""

# Function to get latitude and longitude for candidate address
def get_lat_lon(address):
    try:
        results = geocoder.geocode(address)
        if results and len(results):
            return (results[0]['geometry']['lat'], results[0]['geometry']['lng'])
    except Exception as e:
        print(f"Error: {e}")
    return (None, None)

# Function for similarity search using in-memory data store
def similarity_search(query, k=3, filters=None, candidate_latlon=None, max_distance=None):
    query_embedding = embedding.embed_query(query)
    similarities = []
    for key, value in data_store.items():
        if filters:
            match = all(value.get(filter_key) == filter_value for filter_key, filter_value in filters.items() if filter_value and filter_value != "Select")
            if not match:
                continue
        
        distance = None
        job_latlon = (value['Latitude'], value['Longitude'])
        if candidate_latlon and max_distance and job_latlon != (None, None) and None not in job_latlon:
            try:
                distance = geodesic(candidate_latlon, job_latlon).miles
                if distance > max_distance:
                    continue
            except ValueError:
                continue

        if value.get('License') == 'TRUE' and not license_required:
            continue
        if value.get('FFM') == 'TRUE' and not ffm_required:
            continue

        sim = np.dot(query_embedding, value['embedding']) / (np.linalg.norm(query_embedding) * np.linalg.norm(value['embedding']))
        similarities.append((key, sim, distance if candidate_latlon and max_distance else None))
    
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:k]




# Streamlit app
st.set_page_config(page_title="TalentWorx", layout="centered")

# Add navbar
st.markdown("""
    <style>
        .navbar {
            background-color: transparent;
            padding: 10px;
            margin-top: -50px;  /* Adjust this value to reduce the padding above the navbar */
        }
        .navbar h1 {
            color: white;
            margin: 0;
        }
    </style>
    <nav class="navbar">
        <h1>TalentWorx</h1>
    </nav>
    """, unsafe_allow_html=True)

# Add banner image
# banner_image_path = "img/reddit_background_sm.png"
# banner_image = Image.open(banner_image_path)

# def get_image_base64(image_path):
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# banner_image_base64 = get_image_base64(banner_image_path)

# st.markdown(f"""
#     <div style="display: flex; justify-content: center; width: 75%; margin: auto;">
#         <img src="data:image/png;base64,{banner_image_base64}" style="width: 75%;">
#     </div>
#     """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; width: 75%; margin: auto;">
        <h2>Position Recommendation Tool</h2>
        <p>TalentWorx helps recruiters at Worxweb Solutions match candidates with suitable job roles efficiently. Use the tool below to describe the candidate and apply additional filters to find the best matches.</p>
    </div>
    """, unsafe_allow_html=True)

# Center the form
st.markdown("""
    <div style="display: flex; justify-content: center;">
    <div style="width: 75%;">
    """, unsafe_allow_html=True)

# Form
with st.form('main_form'):
    # Multiline text area for query input
    query = st.text_area("Describe the candidate")

    # File uploader for resume
    resume_file = st.file_uploader("Upload resume (optional)", type=["pdf", "docx"])

    # Candidate address for proximity search
    candidate_address = st.text_input("Enter candidate's address (for proximity search)")

    # Slider for max distance
    max_distance = st.slider("Maximum distance for in-person positions (miles):", min_value=0, max_value=50, value=10)

    # Expandable section for additional filters
    with st.expander("Position filters"):
        status = st.selectbox("Status", options=["Select", "open"] + df['Status'].unique().tolist(), index=1)
        vertical = st.selectbox("Vertical", options=["Select"] + df['Vertical'].unique().tolist(), index=0)
        sub_vertical = st.selectbox("Sub Vertical", options=["Select"] + df['Sub Vertical'].unique().tolist(), index=0)
        functional_expertise = st.selectbox("Functional Expertise", options=["Select"] + df['Functional Expertise'].unique().tolist(), index=0)
        city = st.selectbox("City", options=["Select"] + df['City'].unique().tolist(), index=0)
        type = st.selectbox("Type", options=["Select"] + df['Type'].unique().tolist(), index=0)

    # Candidate has License and FFM
    license_required = st.checkbox("Candidate has License", value=False)
    ffm_required = st.checkbox("Candidate has FFM", value=False)

    # Submit button
    submitted = st.form_submit_button('Search')

    if submitted:
        filters = {
            'Status': status if status != "Select" else None,
            'Vertical': vertical if vertical != "Select" else None,
            'Sub Vertical': sub_vertical if sub_vertical != "Select" else None,
            'Functional Expertise': functional_expertise if functional_expertise != "Select" else None,
            'City': city if city != "Select" else None,
            'Type': type if type != "Select" else None
        }

        # Process resume file if uploaded
        if resume_file:
            resume_text = extract_text_from_file(resume_file)
            query += " " + resume_text

        # Get candidate latitude and longitude if address is provided
        candidate_latlon = get_lat_lon(candidate_address) if candidate_address else None

        results = similarity_search(query, k=3, filters=filters, candidate_latlon=candidate_latlon, max_distance=max_distance)

        # Results display
        st.write("Results:")
        for result in results:
            job = data_store[result[0]]
            distance_str = ""
            job_latlon = (job.get('Latitude'), job.get('Longitude'))
            if candidate_latlon and job_latlon != (None, None) and None not in job_latlon:
                distance = geodesic(candidate_latlon, job_latlon).miles
                distance_str = f" (Distance: {distance:.2f} miles)"
            st.write(f"{job['company']} - {job['position']}, Similarity: {result[1]}{distance_str}")
            st.write(f"Description: {job['description']}")
            st.write(f"City: {job['City']}, Zip: {job['Zip']}, Status: {job['Status']}, Vertical: {job['Vertical']}, Sub Vertical: {job['Sub Vertical']}, Type: {job['Type']}, Functional Expertise: {job['Functional Expertise']}, Physical Address: {job['Physical Address']}")
            st.write("---")

        if not results:
            st.write("No results found.")

st.markdown("</div></div>", unsafe_allow_html=True)
