FROM python:3.12-slim-bullseye

WORKDIR /app

USER root
EXPOSE 8888

# Install linux packages
RUN apt update && apt upgrade -y \
    && apt install libgl1 libglib2.0-0 -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
# Run the 'make poetry-export' command on the host machine to have the file requirements.txt
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy sources
COPY drone_detection/ ./drone_detection
COPY models/ ./models
COPY demo/ ./demo
COPY data/demo_data/ ./data/demo_data
COPY config/ ./config

# Cleaning
RUN rm requirements.txt
RUN pip3 cache purge

#CMD ["start-notebook.sh", "--ServerApp.token=''", "--ServerApp.password=''"]
#CMD ["python", "./demo/inference.py"]
CMD ["tail", "-f", "/dev/null"]
