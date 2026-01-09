FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into /app
COPY . /app/

# CRITICAL: Tell Python to look in /app to find the 'qf_protocol' package
ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Run using the module path. 
# This resolves imports like 'from qf_protocol.core...' correctly.
CMD ["python", "-m", "qf_protocol.api.api_server"]