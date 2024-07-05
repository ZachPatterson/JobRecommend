import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv
from geopy.distance import geodesic
from opencage.geocoder import OpenCageGeocode
import redis
import json
from PyPDF2 import PdfReader
import docx2txt
from langchain.embeddings import OpenAIEmbeddings

# # Set up debugpy for debugging
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

# Load environment variables from .env file
load_dotenv()

# Define weights
SIMILARITY_WEIGHT = 0.8
PROXIMITY_WEIGHT = 0.1
INPERSON_BONUS = 0.1

# Set up Azure OpenAI Embeddings
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME")
MODEL_NAME = os.environ.get("MODEL_NAME")
OPENCAGE_KEY = os.environ.get("OPENCAGE_KEY")
REDIS_ENDPOINT = os.environ.get("REDIS_ENDPOINT")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

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

# Connect to Redis
r = redis.StrictRedis(host=REDIS_ENDPOINT, port=6380, password=REDIS_PASSWORD, ssl=True)

if 'feedback' not in st.session_state:
    st.session_state['feedback'] = {}

# Function to normalize proximity score
def normalize_proximity(distance, max_distance):
    if distance is None:
        return 0
    if max_distance == 0:
        return 1
    return max(0, min(1, 1 - (distance / max_distance)))

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

# Function for similarity search using Redis data store
def similarity_search(query, k=3, filters=None, candidate_latlon=None, max_distance=None, candidate_license=False, candidate_ffm=False):
    query_embedding = embedding.embed_query(query)
    similarities = []

    # Start scanning with an initial cursor value of '0'
    cursor = 0
    while True:  # Begin infinite loop
        cursor, keys = r.scan(cursor=cursor, match='job:*')  # Scan Redis keys
        for key in keys:  # Process each key in the current batch
            value = json.loads(r.get(key))  # Get the value from Redis

            # Check if job requires License or FFM and if candidate meets these requirements
            requires_license = value.get('License')
            requires_ffm = value.get('FFM')

            if (requires_license and not candidate_license) or (requires_ffm and not candidate_ffm):
                continue  # Skip this job if it requires a license/FFM that the candidate doesn't have
            
            # Calculate proximity if applicable
            distance = None
            job_latlon = (value['Latitude'], value['Longitude'])
            is_remote = value['Type'] == 'remote'
            if candidate_latlon and job_latlon != (None, None) and None not in job_latlon and not is_remote:
                try:
                    distance = geodesic(candidate_latlon, job_latlon).miles
                    if distance > max_distance:
                        continue
                except ValueError:
                    continue
            
            # Compute scores and composite score
            job_embedding = np.array(value['embedding'])
            similarity_score = np.dot(query_embedding, job_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(job_embedding))
            proximity_score = normalize_proximity(distance, max_distance) if distance is not None else 0
            job_type_score = INPERSON_BONUS if not is_remote else 0
            composite_score = SIMILARITY_WEIGHT * similarity_score + PROXIMITY_WEIGHT * proximity_score + job_type_score
            
            # Append results
            similarities.append((key, composite_score, similarity_score, proximity_score, distance if candidate_latlon and max_distance else None, is_remote))
        
        # Check if all keys have been scanned (cursor == '0' means end of data)
        if cursor == 0:
            break  # Exit the loop

    # Sort and return the top k results
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:k]


# Function to store feedback in Redis
def store_feedback(query, job_id, feedback):
    query_embedding = embedding.embed_query(query)
    feedback_data = {
        'query': query,
        'candidate_embedding': query_embedding.tolist(),
        'job_id': job_id,
        'feedback': feedback
    }
    r.rpush('feedback', json.dumps(feedback_data))

def get_unique_values(field):
    cache_key = f"unique_values:{field}"
    cached_values = r.get(cache_key)
    if cached_values:
        return json.loads(cached_values)

    cursor = 0
    unique_values = set()
    while True:
        cursor, keys = r.scan(cursor=cursor, match='job:*')
        for key in keys:
            value = json.loads(r.get(key))
            field_value = value.get(field)
            if field_value:
                unique_values.add(field_value)
        if cursor == 0:
            break

    unique_values = sorted(unique_values)
    r.set(cache_key, json.dumps(unique_values), ex=3600)  # Cache for 1 hour
    return unique_values

# Function to adjust embeddings based on feedback
def adjust_embeddings():
    feedback_list = r.lrange('feedback', 0, -1)
    for feedback in feedback_list:
        feedback = json.loads(feedback)
        query_embedding = np.array(feedback['candidate_embedding'])
        job_id = feedback['job_id']
        job_data = json.loads(r.get(f"job:{job_id}"))
        job_embedding = np.array(job_data['embedding'])
        if feedback['feedback'] == 'yes':
            # Decrease distance between embeddings
            job_embedding += 0.01 * (query_embedding - job_embedding)
        elif feedback['feedback'] == 'no':
            # Increase distance between embeddings
            job_embedding -= 0.01 * (query_embedding - job_embedding)
        
        job_data['embedding'] = job_embedding.tolist()
        r.set(f"job:{job_id}", json.dumps(job_data))

    r.delete('feedback')  # Clear feedback after applying
# Get unique values for each filter
unique_status = ["Select"] + get_unique_values('Status')
unique_type = ["Select"] + get_unique_values('Type')
unique_vertical = ["Select"] + get_unique_values('Vertical')
unique_sub_vertical = ["Select"] + get_unique_values('Sub Vertical')
unique_functional_expertise = ["Select"] + get_unique_values('Functional Expertise')
unique_city = ["Select"] + get_unique_values('City')

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
        status = st.selectbox("Status", options=unique_status, index=0)
        type = st.selectbox("Type", options=unique_type, index=0)
        vertical = st.selectbox("Vertical", options=unique_vertical, index=0)
        sub_vertical = st.selectbox("Sub Vertical", options=unique_sub_vertical, index=0)
        functional_expertise = st.selectbox("Functional Expertise", options=unique_functional_expertise, index=0)
        city = st.selectbox("City", options=unique_city, index=0)

    # Candidate has License and FFM
    candidate_license = st.checkbox("Candidate has License", value=False)
    candidate_ffm = st.checkbox("Candidate has FFM", value=False)

    # Submit button
    submitted = st.form_submit_button('Search')

if submitted:
    filters = {
        'Status': status if status != "Select" else None,
        'Vertical': vertical if vertical != "Select" else None,
        'Sub Vertical': sub_vertical if sub_vertical != "Select" else None,
        'Functional Expertise': functional_expertise if functional_expertise != "Select" else None,
        'City': city if city != "Select" else None,
        'Type': type if type != "Select" else None,
    }

    # Process resume file if uploaded
    if resume_file:
        resume_text = extract_text_from_file(resume_file)
        query += " " + resume_text

    # Get candidate latitude and longitude if address is provided
    candidate_latlon = get_lat_lon(candidate_address) if candidate_address else None

if submitted:
    results = similarity_search(query, k=3, filters=filters, candidate_latlon=candidate_latlon, max_distance=max_distance, candidate_license=candidate_license, candidate_ffm=candidate_ffm)

    # Results display handling
    # Your code to display the results here

    # Results display
    st.write("Results:")
    if results:
        for result in results:
            job = json.loads(r.get(result[0]))
            distance = result[4]  # Assuming result[4] holds the distance
            similarity_score = result[2]  # Assuming result[2] holds the similarity score
            proximity_score = result[3]  # Assuming result[3] holds the proximity score
            composite_score = result[1]  # Assuming result[1] holds the composite score

            city = "N/A" if job['Type'] == 'remote' else job['City']
            zip_code = "N/A" if job['Type'] == 'remote' else job['Zip']

            st.write(f"**Company:** {job['company']}")
            st.write(f"**Position:** {job['position']}")
            st.write(f"**Similarity Score:** {similarity_score:.4f}")
            st.write(f"**Proximity Score:** {proximity_score:.4f} (Adjusted)")
            st.write(f"**Composite Score:** {composite_score:.4f}")
            if distance is not None:
                st.write(f"**Distance:** {distance:.2f} miles")
            st.write(f"**Description:** {job['description']}")
            st.write(f"**City:** {city}, **Zip:** {zip_code}")
            st.write(f"**Status:** {job['Status']}, **Vertical:** {job['Vertical']}")
            st.write(f"**Sub Vertical:** {job['Sub Vertical']}")
            st.write(f"**Type:** {job['Type']}, **Functional Expertise:** {job['Functional Expertise']}")
            st.write(f"**Physical Address:** {job['Physical Address']}")

            # Feedback section
            feedback_key = f"feedback_{result[0]}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = None

            col1, col2 = st.columns(2)
            with col1:
                if st.button('👍', key=f"yes_{result[0]}"):
                    st.session_state[feedback_key] = 'yes'
                    store_feedback(query, job['id'], 'yes')
                    st.write("Thank you for your feedback!")
            with col2:
                if st.button('👎', key=f"no_{result[0]}"):
                    st.session_state[feedback_key] = 'no'
                    store_feedback(query, job['id'], 'no')
                    st.write("Thank you for your feedback!")

            # Display current feedback status
            if st.session_state[feedback_key] == 'yes':
                st.write("Feedback: 👍")
            elif st.session_state[feedback_key] == 'no':
                st.write("Feedback: 👎")
            st.write("---")
    else:
        st.write("No results found.")

# # Option to adjust embeddings based on collected feedback
# if st.button("Adjust Embeddings"):
#     adjust_embeddings()
#     st.write("Embeddings adjusted with new feedback.")

st.markdown("</div></div>", unsafe_allow_html=True)
