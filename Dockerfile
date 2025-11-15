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

# Render will pass the $PORT variable.
# The app MUST bind to 0.0.0.0 and use this variable.
CMD /bin/sh -c "uvicorn app:app --host 0.0.0.0 --port $PORT"