def calculate_kpis(current, optimized):
    lead_reduction = (current - optimized) / current

    return {
        "Lead Time Reduction %": lead_reduction * 100,
        "Confidence Score": 0.85,
        "Profit Stability": 0.9
    }