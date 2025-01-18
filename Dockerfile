# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the current directory contents into the container at /app
COPY . /app

# Create and activate the conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "myenv", "/bin/bash", "-c"]

# Fix the deprecated function issue using sed
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /opt/conda/envs/myenv/lib/python3.10/site-packages/basicsr/data/degradations.py

# Set environment variables
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Expose port 7860 for the Gradio app
EXPOSE 7860

# Command to run the Gradio app
CMD ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "app.py"]

