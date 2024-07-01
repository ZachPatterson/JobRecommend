import redis
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

redis_url = f"rediss://:{os.environ['REDIS_PASSWORD']}@{os.environ['REDIS_ENDPOINT']}"

try:
    redis_client = redis.StrictRedis.from_url(redis_url, socket_connect_timeout=20, socket_timeout=10)
    redis_client.ping()
    print("Connected to Redis successfully!")
except redis.exceptions.TimeoutError:
    print("Timeout connecting to Redis")
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
