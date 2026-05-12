import os
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.preprocessing import load_data

def main():

    print("🚀 Starting training...")

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # -----------------------------
    # LOAD DATA
    # -----------------------------

    df = load_data('data/orders.csv')

    print("✅ Data loaded")

    # -----------------------------
    # CREATE REALISTIC DISTANCE
    # -----------------------------

    np.random.seed(42)

    df['distance'] = np.random.randint(
        200,
        1200,
        size=len(df)
    )

    # -----------------------------
    # CREATE REALISTIC LEAD TIME
    # -----------------------------

    df['lead_time'] = (
        (df['distance'] / 250)
        +
        (df['Units'] * 0.2)
        +
        (df['Cost'] / 100)
    )

    # Add slight randomness
    df['lead_time'] += np.random.normal(
        0,
        0.5,
        len(df)
    )

    # Keep realistic values
    df['lead_time'] = df['lead_time'].clip(2, 15)

    print("✅ Lead time created")

    # -----------------------------
    # ENCODE CATEGORICAL COLUMNS
    # -----------------------------

    encoders = {}

    cat_cols = [
        'Ship Mode',
        'Region',
        'Division',
        'Product Name'
    ]

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le

    print("✅ Encoding done")

    # -----------------------------
    # FEATURES
    # -----------------------------

    features = [
        'Ship Mode',
        'Region',
        'Division',
        'Product Name',
        'Sales',
        'Units',
        'Cost',
        'distance'
    ]

    X = df[features]

    y = df['lead_time']

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # MODEL
    # -----------------------------

    model = GradientBoostingRegressor()

    model.fit(X_train, y_train)

    # -----------------------------
    # EVALUATION
    # -----------------------------

    preds = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    print(f"✅ MAE: {mae:.2f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------

    joblib.dump(
        model,
        'models/best_model.pkl'
    )

    joblib.dump(
        encoders,
        'models/encoders.pkl'
    )

    print("✅ Model saved successfully")


if __name__ == "__main__":
    main()