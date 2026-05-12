import pandas as pd
import random

def simulate(df, product, model, encoders):

    # Random row for selected product
    base = df[
        df['Product Name'] == product
    ].sample(1).iloc[0].copy()

    # Factory list
    factories = [
        "Lot's O' Nuts",
        "Wicked Choccy's",
        "Sugar Shack",
        "Secret Factory",
        "The Other Factory"
    ]

    # Different factory distances
    factory_distances = {
        "Lot's O' Nuts": 300,
        "Wicked Choccy's": 550,
        "Sugar Shack": 850,
        "Secret Factory": 450,
        "The Other Factory": 700
    }

    # Factory performance impact
    factory_bonus = {
        "Lot's O' Nuts": -1.0,
        "Wicked Choccy's": 0.5,
        "Sugar Shack": 1.5,
        "Secret Factory": -0.3,
        "The Other Factory": 0.8
    }

    results = []

    for factory in factories:

        temp = base.copy()

        # Encode categorical columns
        categorical_cols = [
            'Ship Mode',
            'Region',
            'Division',
            'Product Name'
        ]

        for col in categorical_cols:

            temp[col] = encoders[col].transform(
                [str(temp[col])]
            )[0]

        # Assign distance
        temp['distance'] = factory_distances[factory]

        # Features
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

        # Input dataframe
        input_df = pd.DataFrame(
            [temp[features]]
        )

        # Predict lead time
        pred = model.predict(input_df)[0]

        # Add factory impact
        pred += factory_bonus[factory]

        # Small randomness
        pred += random.uniform(-0.3, 0.3)

        # Prevent unrealistic values
        pred = max(1, pred)

        # Profit calculation
        base_profit = (
            temp['Sales'] - temp['Cost']
        )

        # Adjusted profit
        adjusted_profit = (
            base_profit + (10 - pred)
            )

        results.append({
            'factory': factory,
            'distance': temp['distance'],
            'lead_time': round(pred, 2),
            'profit': round(adjusted_profit, 2)
        })

    # Create dataframe
    results_df = pd.DataFrame(results)

    return results_df