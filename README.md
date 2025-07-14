# Email-Spam-Detection
# Spam Detection Web App

A simple web application for detecting spam messages using a machine learning model (Naive Bayes) and Flask.

## Features

- Classifies messages as "Spam" or "Not Spam"
- Trained on SMS spam dataset
- Web interface for easy message testing

## Demo

![screenshot](screenshot.png) <!-- Add a screenshot if you have one -->

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/spam-detection-flask-app.git
    cd spam-detection-flask-app
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Train the model (if not already trained):**
    ```sh
    python train_model.py
    ```

4. **Run the app:**
    ```sh
    python app.py
    ```

5. **Open your browser and go to:**
    ```
    http://127.0.0.1:5000/
    ```

## Project Structure

```
├── app.py
├── train_model.py
├── requirements.txt
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── spam_updated.csv
```

## Requirements

See [requirements.txt](requirements.txt) for all dependencies.

## License
This project is licensed under the MIT License.
