import pandas as pd

def simulate(df, product, model, encoders):

    # Get selected product row
    base = df[df['Product Name'] == product].iloc[0].copy()

    # Factory list
    factories = [
        "Lot's O' Nuts",
        "Wicked Choccy's",
        "Sugar Shack",
        "Secret Factory",
        "The Other Factory"
    ]

    # Simulated distances for each factory
    factory_distances = {
        "Lot's O' Nuts": 300,
        "Wicked Choccy's": 550,
        "Sugar Shack": 850,
        "Secret Factory": 450,
        "The Other Factory": 700
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

        # Different distance for each factory
        temp['distance'] = factory_distances[factory]

        # Features used for prediction
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

        # Prepare input dataframe
        input_df = pd.DataFrame([temp[features]])

        # Predict lead time
        pred = model.predict(input_df)[0]

        # Prevent unrealistic values
        pred = max(1, pred)

        # Profit calculation
        base_profit = temp['Sales'] - temp['Cost']

        # Adjust profit based on lead time
        adjusted_profit = base_profit - (pred * 0.5)

        results.append({
            'factory': factory,
            'distance': temp['distance'],
            'lead_time': round(pred, 2),
            'profit': round(adjusted_profit, 2)
        })

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    return results_df