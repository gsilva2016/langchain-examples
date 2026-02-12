#!/bin/bash
set -e  

# Source environment variables
source .env

# Logging convenience functions
log_info() {
    echo "[INFO] $1"
}
log_warn() {
    echo "[WARN] $1"
}
log_error() {
    echo "[ERROR] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare version numbers
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Verifying system requirements
log_info "Installing llama-cpp..."
log_info "Step 1: Verifying system requirements..."

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$NAME" != "Ubuntu" ]]; then
        log_error "Installation requires Ubuntu. Detected: $NAME"
        exit 1
    fi
    
    if [[ "$VERSION_ID" != "24.04" ]]; then
        log_error "Expected Ubuntu 24.04, but detected version $VERSION_ID"
        exit 1
    else
        log_info "Ubuntu 24.04 detected"
    fi
else
    log_error "Cannot detect OS version. /etc/os-release not found."
    exit 1
fi

# Check kernel version
KERNEL_VERSION=$(uname -r | cut -d'-' -f1)
REQUIRED_KERNEL="6.14.0"

log_info "Detected kernel version: $KERNEL_VERSION"

if version_ge "$KERNEL_VERSION" "$REQUIRED_KERNEL"; then
    log_info "Kernel version $KERNEL_VERSION >= $REQUIRED_KERNEL: Success!"
else
    log_error "Kernel version $KERNEL_VERSION is older than required $REQUIRED_KERNEL"
    exit 1
fi

# Check if i915 driver is loaded
if lsmod | grep -q "^i915 "; then
    log_info "i915 kernel driver loaded successfully"
else
    log_error "Failed to load i915 driver. Please check your system configuration."
    exit 1
fi

# Check for required tools
log_info "Checking for required tools..."
for tool in wget tar; do
    if ! command_exists "$tool"; then
        log_warn "$tool is not installed. Attempting to install..."
        if sudo apt install -y "$tool"; then
            log_info "$tool installed successfully"
        else
            log_error "Failed to install $tool. Please install it manually: sudo apt install $tool"
            exit 1
        fi
    fi
done
log_info "Required tools available."

# Create directories
INSTALL_DIR="$LLAMA_CPP_INSTALL_DIR"
MODELS_DIR="$INSTALL_DIR/models"

log_info "Creating installation directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$MODELS_DIR"

cd "$INSTALL_DIR"
MODELS_DIR="./models"

# Download llama-cpp package
log_info "Step 2: Downloading llama-cpp package..."

PACKAGE_URL="$LLAMA_CPP_PACKAGE_URL"
PACKAGE_FILE="$(basename "$PACKAGE_URL")"

if [ -f "$PACKAGE_FILE" ]; then
    log_warn "Package file already exists. Skipping download."
else
    log_info "Downloading from $PACKAGE_URL..."
    if wget -c "$PACKAGE_URL" -O "$PACKAGE_FILE"; then
        log_info "Package downloaded successfully"
    else
        log_error "Failed to download package"
        exit 1
    fi
fi

# Download model files
log_info "Step 3: Downloading model files..."

MODEL_URL="$LLAMA_CPP_MODEL_URL"
MODEL_FILE="$MODELS_DIR/$(basename "$MODEL_URL")"

MMPROJ_URL="$LLAMA_CPP_MMPROJ_URL"
MMPROJ_FILE="$MODELS_DIR/mmproj-F16.gguf"

# Download model file
if [ -f "$MODEL_FILE" ]; then
    log_warn "Model file already exists. Skipping download."
else
    log_info "Downloading model file (this may take a while)..."
    if wget -c "$MODEL_URL" -O "$MODEL_FILE"; then
        log_info "Model file downloaded successfully!"
    else
        log_error "Failed to download model file"
        exit 1
    fi
fi

# Download mmproj file
if [ -f "$MMPROJ_FILE" ]; then
    log_warn "mmproj file already exists. Skipping download."
else
    log_info "Downloading mmproj file..."
    if wget -c "$MMPROJ_URL" -O "$MMPROJ_FILE"; then
        log_info "mmproj file downloaded successfully!"
    else
        log_error "Failed to download mmproj file"
        exit 1
    fi
fi

# Extract llama-cpp package
log_info "Step 4: Extracting llama-cpp package..."

if [ -d "bin" ]; then
    log_warn "bin directory already exists. Skipping extraction."
else
    log_info "Extracting $PACKAGE_FILE..."
    if tar -xzf "$PACKAGE_FILE"; then
        log_info "Package extracted successfully!"
        
        # CD into extracted directory
        EXTRACTED_DIR=$(tar -tzf "$PACKAGE_FILE" | head -1 | cut -d/ -f1)
        cd "$EXTRACTED_DIR"
        log_info "Changed to directory: $EXTRACTED_DIR"
    else
        log_error "Failed to extract package"
        exit 1
    fi
fi

# Verify llama-server executable exists
if [ -f "llama-server" ]; then
    log_info "llama-server executable found"
    chmod +x llama-server
else
    log_error "llama-server executable not found after extraction"
    exit 1
fi

log_info "Installation completed successfully!"