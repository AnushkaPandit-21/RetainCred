def predict_churn(model, X):
    prob = model.predict_proba(X)[:,1]
    return prob
