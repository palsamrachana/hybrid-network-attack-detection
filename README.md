# Hybrid Intelligent Intrusion Detection System (IDS)

A comprehensive, modular intrusion detection system that combines machine learning, rule-based detection, and explainable AI for enhanced cybersecurity threat detection.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each functionality
- **Hybrid Detection**: Combines ML models (Random Forest, SVM) with rule-based detection
- **Explainable AI**: Uses SHAP to provide interpretable explanations for predictions
- **Real-time Processing**: FastAPI backend for efficient real-time threat detection
- **Automated Response**: Configurable mitigation strategies based on threat severity
- **Simple Frontend**: Basic web interface for testing and monitoring
- Live Packet Capture using Scapy

## Tech Stack

- **Backend**: FastAPI with Uvicorn
- **Machine Learning**: Scikit-learn, TensorFlow
- **Explainable AI**: SHAP
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
net_ids/
├── data/                    # Dataset storage and management
├── preprocessing/          # Data preprocessing and feature engineering
├── models/                 # ML model training and management
├── detection/              # Hybrid detection logic
├── explainability/         # SHAP-based explanation generation
├── alerts/                 # Alert management and classification
├── mitigation/             # Automated response and mitigation
├── api/                    # FastAPI endpoints
├── utils/                  # Shared utilities and logging
├── frontend/               # Simple web interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
4. Install Npcap (Required for Live Packet Capture)

Download from:
https://npcap.com/#download

IMPORTANT:
During installation, enable:
"Install Npcap in WinPcap API-compatible Mode"

## Usage

### API Endpoints

- `POST /detect/`: Submit network traffic data for analysis
- `GET /health/`: Check system health
- `GET /frontend/`: Access the simple web interface

### Frontend Interface

Access the web interface at `http://localhost:8000/frontend/` to:
- Submit test data for analysis
- View predictions, confidence scores, and severity levels
- See feature importance explanations

## Modules

### 1. Input Module (`data/`)
- Simulates network traffic data collection
- Supports CICIDS2017 and NSL-KDD datasets
- Accepts user activity logs

### 2. Data Preprocessing Module (`preprocessing/`)
- Feature extraction and selection
- Data normalization using StandardScaler
- Data enrichment and cleaning

### 3. Hybrid Detection Module (`detection/`)
- **ML Models**: Random Forest, SVM
- **Rule-based Detection**: Predefined signatures and thresholds
- **Decision Logic**: Combines ML and rule-based outputs

### 4. Explainability Module (`explainability/`)
- SHAP-based feature importance
- Prediction explanations
- Visual explanations for model decisions

### 5. Alert Management Module (`alerts/`)
- Severity classification (Low/Medium/High)
- Structured alert generation
- Confidence-based alert prioritization

### 6. Automated Response Module (`mitigation/`)
- **Low Severity**: Log and monitor
- **Medium Severity**: Quarantine session
- **High Severity**: Block IP address

## Configuration

The system can be configured through environment variables or configuration files in the `utils/` directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Submit a pull request

## Live Packet Capture Note

This project uses Scapy for real-time packet capture.
On Windows, Npcap must be installed for this feature to work.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Security Note

This is a demonstration system. For production use, ensure:
- Proper authentication and authorization
- Secure data handling
- Regular security audits
- Model retraining with fresh data