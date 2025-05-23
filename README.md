# MNIST Digit Classifier with Streamlit

A fun interactive demo that predicts handwritten digits (0-9) using a custom CNN model.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sauls6-mnist-demo.streamlit.app/)

## Features
- Draw digits with mouse/touch
- Real-time prediction
- Confidence scoring
- Mobile-friendly

## How It Works
1. Draw a digit on the canvas
2. Click "Predict Digit"
3. See the model's prediction

## Running Locally
```bash
git clone https://github.com/sauls6/mnist-streamlit-demo.git
cd mnist-streamlit-demo
pip install -r requirements.txt
streamlit run app.py
