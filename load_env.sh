#!/bin/sh

# Load secrets from the JSON file and export as environment variables
export $(jq -r 'to_entries | .[] | "\(.key)=\(.value)"' /app/local.secrets.json)

# Run the main application
exec "$@"
