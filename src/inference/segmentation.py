def predict_segment(kmeans, X_scaled):
    clusters = kmeans.predict(X_scaled)
    
    segment_map = {
        0: "Low Engagement Users",
        1: "Credit Revolvers",
        2: "High Value Spenders"
    }
    
    return [segment_map[c] for c in clusters]
