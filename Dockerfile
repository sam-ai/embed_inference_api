FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Use Python 3.11 for better Python perf
# Update the package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev

# Set Python 3.11 as the default version (for python3)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Download get-pip.py script
RUN apt install curl -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip for Python 3.11
RUN python3 get-pip.py

# Verify Python and pip versions
RUN python3 --version && pip3.11 --version

# Set pip3.11 as the default pip command
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/lib/python3.11/dist-packages/pip 1

ENV PYTHONUNBUFFERED=1

# Install necessary dependencies
# RUN apt-get update && \
#     apt-get install -y python3-pip

### Set up user with permissions
# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user


# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory. /app is mounted to the container with -v, 
# but we want to have the right cwd for uvicorn command below
RUN mkdir $HOME/app
# WORKDIR /app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# # Copy the app code and requirements filed
# COPY . /app
# COPY requirements.txt .
# WORKDIR $PYSETUP_PATH
COPY ./requirements.txt  $HOME/app


COPY ./utils $HOME/app/utils
# COPY ./static /app/static
# COPY ./templates /app/templates
COPY ./app.py $HOME/app/app.py
COPY ./download.py $HOME/app/download.py

WORKDIR $HOME/app


# Install the app dependencies
# RUN pip3 install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
        pip3 install -r requirements.txt

### Update permissions for the app
USER root
RUN chmod 777 ~/app/*
USER user

# Expose the FastAPI port
EXPOSE 7860

# Start the FastAPI app using Uvicorn web server
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "14000", "--limit-concurrency", "1000"]
# RUN python3 download.py

# RUN chmod 755 models

CMD ["python3", "app.py", "--host=0.0.0.0", "--port=7860", "--model_path=BAAI/bge-small-en-v1.5", "--num_workers=2"]