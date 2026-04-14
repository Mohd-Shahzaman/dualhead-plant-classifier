# Dual-Head Plant Disease Classifier

A professional agricultural diagnostic system built with Flask and TensorFlow. This application utilizes a dual-head neural network architecture to simultaneously identify plant species and detect specific diseases from leaf images.

## Overview

The system is designed to provide rapid field diagnostics for agricultural health monitoring. By employing a modified EfficientNet backbone with two distinct classification heads, the model achieves high accuracy in identifying both the host plant and the pathological condition. The backend manages user authentication and provides localized treatment recommendations including organic and chemical interventions.

## Core Features

- Dual-Head Inference: Simultaneous prediction of plant species and disease category.
- Comprehensive Remedies: Targeted treatment protocols for identified conditions.
- User Management: Integrated SQLite-based authentication system.
- Secure Processing: Sanitized file handling and validated inference pipeline.

## Technical Stack

- Backend: Flask, Python 3.
- Machine Learning: TensorFlow, Keras, EfficientNet.
- Database: SQLite3.
- Frontend: Vanilla JavaScript, HTML5, CSS3.

## Installation

### Prerequisites
- Python 3.8 or higher.
- TensorFlow-compatible environment.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Mohd-Shahzaman/dualhead-plant-classifier.git
   cd dualhead-plant-classifier
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Execution

1. Start the server:
   ```bash
   python app.py
   ```

2. Access the application at:
   `http://localhost:5000`

## Model Architecture

The classification engine uses a transfer learning approach with an EfficientNet base. The network architecture consists of two specialized output branches:
- **Plant Head**: Categorizes the image into recorded plant species.
- **Disease Head**: Identifies the specific disease present on the foliage.

Predictions are cross-referenced with the internal remedies repository to provide detailed treatment protocols.
