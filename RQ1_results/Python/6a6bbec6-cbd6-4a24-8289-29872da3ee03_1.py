from scipy.special import beta

def compute_bayes_factor(alpha_0, beta_0, alpha_t, beta_t, alpha_c, beta_c, s_t, n_t, s_c, n_c):
    # Probability of data under H0
    prob_data_H0 = beta(alpha_0 + s_t + s_c, beta_0 + n_t + n_c - s_t - s_c) / beta(alpha_0, beta_0)
    
    # Probability of data under H1 for treatment group
    prob_data_H1_treatment = beta(alpha_t + s_t, beta_t + n_t - s_t) / beta(alpha_t, beta_t)
    
    # Probability of data under H1 for control group
    prob_data_H1_control = beta(alpha_c + s_c, beta_c + n_c - s_c) / beta(alpha_c, beta_c)
    
    # Joint probability of data under H1
    prob_data_H1 = prob_data_H1_treatment * prob_data_H1_control
    
    # Compute Bayes Factor
    BF_10 = prob_data_H1 / prob_data_H0
    
    return BF_10

# Example usage:
alpha_0, beta_0 = 1, 1 # Hyperparameters for H0
alpha_t, beta_t = 1, 1 # Hyperparameters for treatment group under H1
alpha_c, beta_c = 1, 1 # Hyperparameters for control group under H1
s_t, n_t = 40, 100  # Successes and total observations for treatment group
s_c, n_c = 30, 100  # Successes and total observations for control group

BF_10 = compute_bayes_factor(alpha_0, beta_0, alpha_t, beta_t, alpha_c, beta_c, s_t, n_t, s_c, n_c)
print(f"Bayes Factor (BF_10) = {BF_10}")
