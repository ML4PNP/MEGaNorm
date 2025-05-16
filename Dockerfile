# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and pip dependencies
RUN pip install --no-cache-dir --upgrade pip \
    jupyterlab \
    matplotlib \
    ipykernel

# Copy MEGaNorm project files into the container
COPY . /app

# Install the MEGaNorm package and its dependencies
RUN pip install --no-cache-dir .

# Expose the Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.open_browser=True", "--allow-root"]
