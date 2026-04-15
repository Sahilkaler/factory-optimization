import pandas as pd

def simulate(df, product, model, encoders):
    base = df[df['Product Name'] == product].iloc[0].copy()

    factories = [
        "Lot's O' Nuts",
        "Wicked Choccy's",
        "Sugar Shack",
        "Secret Factory",
        "The Other Factory"
    ]

    results = []

    for factory in factories:
        temp = base.copy()

        # Encode categorical columns (FIX)
        for col in ['Ship Mode', 'Region', 'Division', 'Product Name']:
            temp[col] = encoders[col].transform([str(temp[col])])[0]

        # Dummy distance (can improve later)
        temp['distance'] = 1000

        features = ['Ship Mode', 'Region', 'Division', 'Product Name', 'Sales', 'Units', 'Cost', 'distance']

        pred = model.predict([temp[features]])[0]

        profit = temp['Sales'] - temp['Cost']
        adjusted_profit = profit - (pred * 0.5)

        results.append({
            'factory': factory,
            'lead_time': round(pred, 2),
            'profit': round(adjusted_profit, 2)
        })

    return pd.DataFrame(results)