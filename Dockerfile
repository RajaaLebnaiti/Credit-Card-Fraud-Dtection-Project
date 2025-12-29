# Base Image Python
FROM python:3.10-slim

# The working directory
WORKDIR app/

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/list/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy all project files
COPY . .

# Expose port for deployment 
EXPOSE 7860


# Command to start the app 
CMD ["python", "deploy/app.py"]

