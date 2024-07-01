import json
import os

# Load secrets from the JSON file
with open('local.secrets.json') as f:
    secrets = json.load(f)

# Set environment variables
for key, value in secrets.items():
    os.environ[key] = value

# Optionally, save the environment variables to a .env file for dotenv to read
with open('.env', 'w') as f:
    for key, value in secrets.items():
        f.write(f"{key}={value}\n")
