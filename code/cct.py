# cct.py
#
# NAME:        Evelyn Jiang
# EMAIL:       jianger@uci.edu
# STUDENT ID:  63947706

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt



def load_data(url):
    df = pd.read_csv(url)
    data = df.iloc[:, 1:].values  # Exclude the first column (Informant ID)
    return data

def perform_inference(data):
    N = data.shape[0]  # Number of informants
    M = data.shape[1]  # Number of items (questions)
    print("N is the number of imformants = " + str(N))  # debug
    print("M is the number of items (questions) = " + str(M))  # debug

    with pm.Model() as model:
        # Define Priors
        D = pm.Uniform("D", 0.5, 1, shape=N)  # Competence levels of informants
        Z = pm.Bernoulli("Z", 0.5, shape=M)   # Consensus answers
        
        # Compute probabilities
        D_reshaped = D[:, None]  # Reshape for broadcasting
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

        # Define Likelihood
        X_obs = pm.Bernoulli("X_obs", p, observed=data)

        # Perform Inference
        trace = pm.sample(2000, chains=4, tune=1000, return_inferencedata=True)
    
    return trace

def analyze_results(data, trace):
    # Convergence diagnostics
    print("\n### Analyze results: exam the convergence diagnostic ###\n")
    print(az.summary(trace, var_names=['D', 'Z'])) 
    
    # visual inspection of chains
    az.plot_trace(trace, var_names=['D', 'Z'])
    plt.show()

    # Informant Competence
    D_posterior = trace.posterior['D'].mean(dim=['chain', 'draw']).values
    print("\nInformant Competence:")
    for i, competence in enumerate(D_posterior):
        print(f"\t Informant D{i + 1}: {competence:.3f}")

    # Visualization of competence
    az.plot_posterior(trace, var_names=['D'])
    plt.show()

    # Consensus Answers
    Z_posterior_prob = trace.posterior['Z'].mean(dim=['chain', 'draw']).values 
    Z_posterior_answer = np.round(Z_posterior_prob)  # Round to get 0 or 1
    print("\nConsensus Answers:")
    for j, answer in enumerate(Z_posterior_answer):
        print(f"\t Item Z{j + 1}: {int(answer)}")

    # Visualization of consensus answers
    az.plot_posterior(trace, var_names=['Z'])
    plt.show()

    # Compare with Naive Aggregation
    # Majority vote
    majority_vote = np.round(data.mean(axis=0))
    print("\nMajority Vote Answers:")
    print(majority_vote.astype(int))

    print("\nComparison (CCT Model vs . Majority Vote):")
    M = data.shape[1]  # Number of items (questions)
    for j in range(M):
        print(f"\t Item {j + 1}: CCT = {int(Z_posterior_answer[j])}, Majority Vote = {int(majority_vote[j])}")


def run():
    # Step 1: Load data
    data = load_data('https://raw.githubusercontent.com/joachimvandekerckhove/cogs107s25/refs/heads/main/1-mpt/data/plant_knowledge.csv')
    print(data)  # debug

    # Step 2: Implement the model and perform inference
    trace = perform_inference(data)

    # Step 3: Analyze Results
    analyze_results(data, trace)


if __name__ == "__main__":
    run()















