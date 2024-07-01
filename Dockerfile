# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Copy the current directory contents into the container at /app
COPY . .

# Copy the startup script into the container
COPY load_env.sh /app/load_env.sh

# Make the startup script executable
RUN chmod +x /app/load_env.sh

# Use the startup script to set environment variables and run the app
ENTRYPOINT ["/app/load_env.sh"]
CMD ["streamlit", "run", "app.py"]
