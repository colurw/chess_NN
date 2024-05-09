# Dockerfile

# Use official Python runtime as a parent image
FROM python:3.10-slim

# Allow docker to cache installed dependencies between builds
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless

# Mount the application code to the image
COPY . code
WORKDIR /code

# Expose ports
EXPOSE 8000

# Run production server
ENTRYPOINT ["python", "django_wrapper/manage.py"]
CMD ["runserver", "0.0.0.0:8000"]