# 🏠 AI House Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

An intelligent deep learning application that predicts house prices in the Greater Seattle area. Built with PyTorch and Streamlit, this system provides accurate market value estimates based on property features.

## ✨ Features

- **Instant Price Estimates**: Get real-time house price predictions
- **Comprehensive Analysis**: Considers location, size, rooms, condition, view, waterfront, age, and renovations
- **Smart Renovation Handling**: Properly accounts for renovation impact on property value
- **Market Comparisons**: Compare estimates against city averages and view price per square foot
- **Confidence Ranges**: See price ranges with 90-110% confidence intervals
- **GPU Accelerated**: Automatically uses NVIDIA RTX 3070 for faster training
- **Clean UI**: User-friendly Streamlit interface with real-time updates

## 🧠 Model Architecture

The system uses a deep neural network with:
- **Input Layer**: 12+ engineered features
- **Hidden Layers**: 256 → 128 → 64 → 32 neurons
- **Batch Normalization**: After each hidden layer
- **Dropout**: 20% regularization to prevent overfitting
- **Activation**: ReLU for non-linearity
- **Output**: Single neuron with log-transformed target

### Features Used:
- Location (city factor)
- Living area (sq ft)
- Bedrooms and bathrooms
- Condition rating (1-5)
- View rating (0-4)
- Waterfront status
- Year built
- Renovation year
- Engineered features (house age, years since renovation, renovation benefit)

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support

## Author
Dr. Mahroona Laraib
