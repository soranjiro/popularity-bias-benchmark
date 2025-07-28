# Benchmarking Popularity Bias in Recommender Systems

This repository contains the official PyTorch and C++ implementation for the paper: "Benchmarking Popularity Bias in Recommender Systems: A Comprehensive Evaluation across Diverse Datasets".

Our work investigates how popularity bias impacts recommendation performance across various datasets. We find that its effect is not uniform; it can be beneficial or harmful depending on the dataset's unique characteristics

## Key Features
Debiasing Methods: Implementation of novel and existing debiasing methods focusing on review count and average rating (TIDE, SbC, aSbC, qaSbC).
Comprehensive Benchmarking: Scripts to run experiments on 10 different datasets to reproduce the paper's results.
Backbone Models: Support for multiple backbone models, including LightGCN, MF, and NCF.
Dataset-Specific Analysis: Tools to analyze how popularity signals (quality vs. conformity) differ across datasets.

## Usage

### Prerequisites


```bash
conda env create -f environment.yaml

python main.py
```

### Usage with Docker

To run the project using Docker, follow these steps:

1. **Build the Docker image**
   Navigate to the project directory and build the Docker image:

   ```bash
   docker build -t <image_name> .
   ```

   Replace `<image_name>` with a name of your choice (e.g., `recommender_system`).

2. **Run the Docker container**
   Start the container with GPU support:

   ```bash
   docker run --rm -it --gpus all -v $(pwd):/app -w /app <image_name>
   ```

   Replace `<image_name>` with the name you used during the build step.

3. **Inside the container**
   The container will automatically activate the conda environment and execute `main.py`. If you need to run other scripts or commands, you can modify the `CMD` instruction in the Dockerfile or execute commands interactively inside the container.

**Note:** Ensure that you have NVIDIA drivers and `nvidia-container-toolkit` installed on your system to enable GPU support.
