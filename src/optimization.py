def optimize(df, weight_time=0.6, weight_profit=0.4):
    max_time = df['lead_time'].max()

    df['time_score'] = max_time - df['lead_time']
    df['profit_score'] = df['profit']

    df['final_score'] = (
        weight_time * df['time_score'] +
        weight_profit * df['profit_score']
    )

    return df.sort_values(by='final_score', ascending=False)