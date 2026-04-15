import os
import joblib
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import load_data
from src.modeling import train_models

def main():
    print("🚀 Starting training...")

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Load data
    df = load_data('data/orders.csv')
    print("✅ Data loaded")

    # Encode categorical columns
    encoders = {}
    cat_cols = ['Ship Mode', 'Region', 'Division', 'Product Name']

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print("✅ Encoding done")

    # Add distance column
    df['distance'] = 1000

    # Save encoders
    joblib.dump(encoders, 'models/encoders.pkl')
    print("✅ Encoders saved")

    # Train model
    model = train_models(df)

    print("✅ Training Complete")

if __name__ == "__main__":
    main()