# Use the ROCm PyTorch image
FROM rocm/pytorch:latest

# Set the working directory
WORKDIR /workspace

# Install Python packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0

# Install required Python libraries
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm4.2
RUN pip install pandas Pillow tqdm pycocotools scikit-learn numpy==1.22.4

# Install torchvision with ROCm support if not already included
# Ensure torch, torchvision, and torchaudio versions are compatible
# Sometimes you might need to install a specific version depending on compatibility with ROCm
# This command assumes ROCm compatibility; adjust as necessary for your ROCm version

# Copy your project files into the container (if desired)
# COPY . /workspace

# Set the default command to keep the container running (this is useful for development)
CMD ["sleep", "infinity"]
