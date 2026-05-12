def optimize(df, weight_time=0.6, weight_profit=0.4):

    df = df.copy()

    # Normalize lead time
    max_time = df['lead_time'].max()
    min_time = df['lead_time'].min()

    df['time_score'] = (
        max_time - df['lead_time']
    ) / (max_time - min_time + 1e-6)

    # Normalize profit
    max_profit = df['profit'].max()
    min_profit = df['profit'].min()

    df['profit_score'] = (
        df['profit'] - min_profit
    ) / (max_profit - min_profit + 1e-6)

    # Final weighted score
    df['final_score'] = (
        weight_time * df['time_score']
        +
        weight_profit * df['profit_score']
    )

    # Sort best first
    return df.sort_values(
        by='final_score',
        ascending=False
    )