# src/dds_ga.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Simulate a genetic algorithm that selects optimal features

def fitness_function(X_train, y_train, X_val, y_val, feature_mask):
    selected_features = X_train.columns[feature_mask == 1]
    if len(selected_features) == 0:
        return 0
    model = RandomForestClassifier()
    model.fit(X_train[selected_features], y_train)
    preds = model.predict(X_val[selected_features])
    return accuracy_score(y_val, preds)

def genetic_algorithm(X, y, generations=10, population_size=8):
    num_features = X.shape[1]
    population = np.random.randint(0, 2, size=(population_size, num_features))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for generation in range(generations):
        scores = [fitness_function(X_train, y_train, X_val, y_val, individual) for individual in population]
        print(f"Generation {generation + 1}, Best Accuracy: {max(scores):.4f}")
        sorted_indices = np.argsort(scores)[::-1]
        population = population[sorted_indices]
        # Simple crossover and mutation
        next_gen = [population[0], population[1]]  # Elitism
        while len(next_gen) < population_size:
            p1, p2 = population[np.random.choice(4)], population[np.random.choice(4)]
            crossover = np.random.randint(1, num_features - 1)
            child = np.concatenate((p1[:crossover], p2[crossover:]))
            mutation = np.random.randint(0, num_features)
            child[mutation] = 1 - child[mutation]
            next_gen.append(child)
        population = np.array(next_gen)

    best_individual = population[0]
    best_features = X.columns[best_individual == 1]
    print("Best Selected Features:", list(best_features))

    # Final model training
    final_model = RandomForestClassifier()
    final_model.fit(X[best_features], y)
    return final_model, best_features

# Example usage (not executed automatically)
if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("../data/DrivingDataset.csv")
    X = data.drop(columns=["labels", "trajectory_id", "start_time", "end_time"])
    y = data["labels"]
    model, features = genetic_algorithm(X, y)
    import joblib
    joblib.dump(model, "../models/dds_model.pkl")
