FROM python:3.12

WORKDIR /app

USER root
EXPOSE 8888

# Update packages
RUN apt update && apt upgrade -y \
    && apt install libgl1 -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip3 install 'poetry==1.8.3'
COPY pyproject.toml .
COPY poetry.toml .
COPY poetry.lock .
COPY Makefile .

# Install packages
RUN make project-init
#RUN make poetry-export
#RUN pip3 install --no-cache-dir --upgrade pip && \
#    pip3 install --no-cache-dir -r requirements.txt

# Copy sources
COPY drone_detection/ ./drone_detection
COPY models/ ./models
COPY demo/ ./demo
COPY data/demo_data/ ./data/demo_data
COPY config/ ./config
COPY README.md .

# Cleaning
RUN make poetry-clear
#RUN rm requirements.txt pyproject.toml README.md poetry.lock Makefile

# Set permissions
#RUN chown ${NB_UID}:${NB_GID} -R /app
# Fix after poetry install
#RUN chown ${NB_UID}:${NB_GID} -R /home/jovyan/.local

# Return user
#USER ${NB_UID}

#CMD ["start-notebook.sh", "--ServerApp.token=''", "--ServerApp.password=''"]
#CMD ["python", "./demo/inference.py"]
CMD ["tail", "-f", "/dev/null"]
