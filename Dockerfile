# Use a standard Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all other application files (app.py, .onnx, .pkl, etc.)
COPY . /code/

# Expose the port Hugging Face Spaces expects (Gradio/Streamlit default)
# For FastAPI, 7860 is standard, but 8000 works if configured. We'll use 7860.
EXPOSE 7860

# Command to run the FastAPI app with uvicorn
# It must bind to 0.0.0.0 to be accessible.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]