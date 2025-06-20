# Bin2Vul Binary Vulnerability Analysis System

## Project Overview
Bin2Vul is an advanced binary code analysis and vulnerability detection system built on the Django framework. By integrating deep learning technologies (ASM2VEC and BERT), it provides security researchers, reverse engineers, and software developers with a powerful platform for binary code analysis.

### Core Features
- 🔍 **Binary Similarity Analysis**
  - Support for binary formats (ELF, PE)
  - Deep learning-based code semantic analysis
  - Visualized similarity comparison results
- 🛡️ **Vulnerability Detection**
  - CWE vulnerability type identification
  - Vulnerability feature extraction and matching
  - Detailed vulnerability reporting
- 📊 **Analysis Report Management**
  - Historical analysis tracking
  - Batch analysis support
  - Report export functionality

## Technical Architecture

### Backend Technology
- **Core Framework**: Django 4.2
  - RESTful API implementation
  - Django ORM for database operations
  - Django template engine for server-side rendering
- **Deep Learning Engine**:
  - PyTorch 2.1.1 for model inference
  - ASM2VEC model for binary code embedding
  - BERT model for vulnerability detection
- **Data Storage**:
  - SQLite3 for development environment
  - Binary file storage with secure access control
- **Asynchronous Processing**:
  - Background task handling for long-running analyses
  - Progress tracking and status updates

### Frontend Technology
- **Core Framework**:
  - Bootstrap 5.1 for responsive UI
  - JavaScript/jQuery for dynamic interactions
- **Visualization**:
  - D3.js for binary similarity visualization
  - Chart.js for analysis statistics
- **Key Features**:
  - Real-time analysis progress display
  - Interactive binary comparison view
  - Responsive dashboard design
  - Dark/Light theme support

### API Integration
- RESTful API endpoints for:
  - Binary upload and analysis
  - Similarity comparison
  - Vulnerability detection
  - Report generation
- Swagger/OpenAPI documentation
- Authentication and rate limiting

## System Requirements
- Python 3.8+
- 8GB+ RAM
- CUDA support (recommended for GPU acceleration)
- Modern browsers (Chrome 90+/Firefox 88+/Edge 90+)

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. System Configuration
```bash
# Create .env file
cp .env.example .env

# Initialize database
python manage.py migrate

# Create admin account
python manage.py createsuperuser
```

### 3. Launch Service
```bash
# Development environment
python manage.py runserver
```

## Detailed Features

### Binary Similarity Analysis
1. Supported File Formats:
   - Windows: EXE
   - Linux: ELF

2. Analysis Features:
   - Function-level similarity comparison

3. Output Results:
   - Function-specific similarity scores (0-100%)
   - Function mapping relationships

### Vulnerability Detection
1. Supported Vulnerability Types:
   - Buffer overflow
   - Integer overflow
   - Format string vulnerability
   - Null pointer dereference
   - Memory leak

2. Detection Method:
   - Deep learning inference

3. Report Contents:
   - Vulnerability type and CWE number
   - Vulnerable function identification
   - Remediation suggestions

## Performance Optimization
1. Hardware Configuration:
   - CPU: 8+ cores
   - RAM: 16GB+
   - GPU: NVIDIA GPU with 8GB+

2. System Settings:
   - Use SSD storage
   - Enable CUDA support
   - Adjust worker processes

## Troubleshooting
1. Model Loading Failure
   ```bash
   # Check model files
   python manage.py check_models
   ```

2. Memory Issues
   ```bash
   # Adjust batch size
   vim bin2src_web/settings.py
   # Modify BATCH_SIZE parameter
   ```

3. Database Migration Issues
   ```bash
   # Reset migrations
   python manage.py migrate --fake-initial
   ```

## Security Guidelines
- Regular dependency updates
- Environment variable management with .env
- File upload size restrictions
- Access control implementation

## License
This project is licensed under the MIT License. See the [LICENSE] file for details. 
