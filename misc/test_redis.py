import redis
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up Redis
redis_endpoint = os.getenv("REDIS_ENDPOINT")
redis_password = os.getenv("REDIS_PASSWORD")

# Connect to Redis
try:
    r = redis.StrictRedis(host=redis_endpoint, port=6380, password=redis_password, ssl=True)
    r.ping()
    print("Connected to Redis successfully!")
except redis.exceptions.AuthenticationError:
    print("Invalid username-password pair.")
except Exception as e:
    print(f"An error occurred: {e}")
