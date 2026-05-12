# Factory Optimization System

A machine learning-based supply chain optimization system that predicts shipping lead time, simulates factory allocation scenarios, and recommends the most efficient factory based on delivery performance and profit.

---

## Features

- Shipping lead time prediction using Machine Learning
- Factory assignment simulation
- Optimization based on lead time and profit
- Interactive Streamlit dashboard
- Data visualization and comparison charts

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## Project Structure

```bash
factory_optimization/
│
├── app/
│   └── app.py
│
├── data/
│   └── orders.csv
│
├── models/
│   ├── best_model.pkl
│   └── encoders.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── simulation.py
│   └── optimization.py
│
└── requirements.txt
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Sahilkaler/factory-optimization
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app/app.py
```

---

## How It Works

1. Load and preprocess historical order data
2. Train machine learning models
3. Predict shipping lead time
4. Simulate multiple factory allocation scenarios
5. Optimize and recommend the best factory

---

## Future Improvements

- Real-time logistics data integration
- Advanced optimization algorithms
- Improved dashboard UI
- Support for larger datasets

---
