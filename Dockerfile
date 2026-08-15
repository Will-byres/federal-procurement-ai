
# 1. Set the Base Image (OS + Python version)
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code and data into the container
COPY . .

# 6. Inform Docker that the container will listen on Streamlit's default port
EXPOSE 8501

# 7. Set the command to execute when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]