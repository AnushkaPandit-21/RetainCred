def recommend_offer(row):

    # HIGH RISK → retention offers
    if row['risk_level'] == "High Risk":
        
        if row['segment'] == "Low Engagement Users":
            return "Annual fee waiver + bonus reward points"
        
        if row['segment'] == "Credit Revolvers":
            return "No-cost EMI + balance transfer offer"
        
        if row['segment'] == "High Value Spenders":
            return "Premium travel cashback + lounge access"

    # MEDIUM RISK → engagement offers
    if row['risk_level'] == "Medium Risk":
        
        if row['spend_level'] == "High Spend":
            return "5x reward points on all purchases"
        
        if row['txn_level'] == "Low Activity":
            return "10% cashback on groceries & fuel"
        
        return "15% cashback on dining"

    # LOW RISK → upsell / loyalty offers
    if row['risk_level'] == "Low Risk":
        
        if row['segment'] == "High Value Spenders":
            return "Pre-approved credit limit increase"
        
        if row['segment'] == "Credit Revolvers":
            return "Lower interest rate upgrade"
        
        return "Bonus reward points booster"