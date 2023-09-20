# Use an official Python runtime with CUDA support as a parent image
FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3.8 python3-pip

# Set the working directory in the container
WORKDIR /app

# Install FastAI and its dependencies
RUN pip3 install fastai
RUN pip install jupyter
RUN pip install jupytext
RUN pip install fastbook


# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
