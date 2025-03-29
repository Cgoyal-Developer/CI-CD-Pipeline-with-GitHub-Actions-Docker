FROM python:3.10-slim AS build_image

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone the Titanic-Project repository
RUN git clone https://github.com/Cgoyal-Developer/Titanic-Project.git /Titanic-Project

# Set working directory to the Titanic-Project directory
WORKDIR /Titanic-Project

# Use a new base image for running the app
FROM python:3.10-slim

# Set the working directory to /app inside the container
WORKDIR /app

# Copy the project files and the installed libraries from the build image to the final image
COPY --from=build_image /Titanic-Project /app

# Install the Python dependencies again in the final image (important to ensure everything is installed)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port for Streamlit (default is 8501)
EXPOSE 8501

# Ensure app.py exists and is in the correct directory before starting Streamlit
CMD ["streamlit", "run", "app.py"]
