FROM python:3.12-slim-bullseye

WORKDIR /app

USER root
EXPOSE 8888

RUN apt update && apt upgrade -y \
    && apt install libgl1 -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.toml poetry.lock Makefile README.md ./

RUN pip3 install 'poetry==1.8.3'
RUN make project-init

# Copy sources
COPY drone_detection/ ./drone_detection
COPY models/ ./models
COPY demo/ ./demo
COPY data/demo_data/ ./data/demo_data
COPY config/ ./config

RUN make clean-all

#CMD ["start-notebook.sh", "--ServerApp.token=''", "--ServerApp.password=''"]
#CMD ["python", "./demo/inference.py"]
CMD ["tail", "-f", "/dev/null"]
