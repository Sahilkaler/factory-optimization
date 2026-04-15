from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import numpy as np

def train_models(df):
    features = ['Ship Mode', 'Region', 'Division', 'Product Name', 'Sales', 'Units', 'Cost', 'distance']
    target = 'lead_time'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Linear": LinearRegression(),
        "RF": RandomForestRegressor(),
        "GB": GradientBoostingRegressor()
    }

    best_model = None
    best_score = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"{name} -> RMSE:{rmse:.2f}, MAE:{mae:.2f}, R2:{r2:.2f}")

        if rmse < best_score:
            best_score = rmse
            best_model = model

    joblib.dump(best_model, 'models/best_model.pkl')

    return best_model