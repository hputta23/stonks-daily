# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (Render/Heroku set PORT env var, but 8000 is default)
ENV PORT=8000
EXPOSE 8000

# Command to run the application
# We use shell form to allow variable expansion for $PORT
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
