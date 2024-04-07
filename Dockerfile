# Install base Python image
FROM python:3.8-slim-buster


# Copies your code file, 'requirements.txt' from 
# your local directory to the filesystem of the container at the path '/app'.
COPY *.py /app/
COPY requirements.txt /app/

# Set the working directory to '/app' in the Docker image.
WORKDIR /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port uvicorn will listen on
EXPOSE 80

# Run uvicorn server on startup
CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
