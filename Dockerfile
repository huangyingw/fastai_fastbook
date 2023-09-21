# Use an official Python runtime with CUDA support as a parent image
FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Install Python, pip, and Graphviz
RUN apt-get update && \
        apt-get install -y python3.8 python3-pip graphviz && \
        rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install FastAI and its dependencies
RUN python3 -m pip install --no-cache-dir fastai jupyter jupytext fastbook

# Copy requirements file and install requirements
COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
