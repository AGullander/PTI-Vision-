# Setup Guide for Container Inspection System

## Overview
This system uses AI models optimized with OpenVINO for container inspection and analysis. Follow the instructions below to set up the required dependencies and download the necessary models.

## Installing OpenVINO

OpenVINO is required to run the optimized AI models. Follow these steps:

1. Visit the official OpenVINO installation guide:  
   https://docs.openvino.ai/2024/get-started/install-openvino.html

2. Choose your platform (Windows, Linux, macOS).

3. Install using pip (recommended for Python users):  
   ```bash
   pip install openvino
   ```

4. Alternatively, download the full OpenVINO toolkit from the website for more features.

5. Verify installation:  
   ```bash
   python -c "import openvino; print('OpenVINO version:', openvino.__version__)"
   ```

## Downloading and Setting Up Models

The system uses the following pre-optimized OpenVINO models. The optimized versions are already included in the `models/` directory, but here's how to download the original models if needed:

### MiniCPM-V (Multimodal Vision-Language Model)
- **Purpose**: Vision-language understanding for container inspection analysis
- **Download Link**: https://huggingface.co/openbmb/MiniCPM-V-2_6
- **Size**: ~2.6GB
- **Use Case**: Analyzes images and provides detailed descriptions of container conditions

**Download Command**:
```bash
pip install huggingface_hub
huggingface-cli download openbmb/MiniCPM-V-2_6 --local-dir models/MiniCPM-V
```

### TinyLlama (Language Model)
- **Purpose**: Text generation and understanding for inspection reports
- **Download Link**: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Size**: ~1.1B parameters
- **Use Case**: Generates human-readable inspection summaries and recommendations

**Download Command**:
```bash
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama
```

### YOLOv8 Model (Object Detection)
- **Purpose**: Detects objects, damage, and anomalies in container images
- **Download Link**: https://github.com/ultralytics/ultralytics
- **Use Case**: Identifies containers, doors, people, and various types of damage

**Download Command**:
```bash
pip install ultralytics
yolo export model=yolov8n.pt format=openvino
```

## Model Optimization (Optional)

If you need to optimize additional models for OpenVINO:

1. Install OpenVINO Development Tools:  
   ```bash
   pip install openvino-dev
   ```

2. Convert a model to OpenVINO IR format:  
   ```bash
   mo --input_model model.onnx --output_dir optimized_model
   ```

The pre-optimized models in this project are already in the correct format.

## Running the Application

1. **Backend Setup**:
   - Ensure Python 3.8+ is installed
   - Install requirements: `pip install -r requirements.txt`
   - Run the backend: `python app.py`

2. **Frontend Setup**:
   - Navigate to frontend directory: `cd frontend`
   - Install dependencies: `pnpm install`
   - Run development server: `pnpm dev`
   - Open http://localhost:3000

3. **Access the Application**:
   - Upload container inspection videos/images
   - Use the analysis tools for automated inspection
   - Review generated reports and recommendations

## System Requirements

- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space for models and data
- **GPU**: Optional, but recommended for faster inference

## Troubleshooting

- **OpenVINO Import Error**: Ensure OpenVINO is properly installed and added to PATH
- **Model Loading Issues**: Verify model files are in the correct `models/` directory
- **Performance Issues**: Check system resources and consider GPU acceleration

## Support

For technical support or questions about model setup, please contact the development team.