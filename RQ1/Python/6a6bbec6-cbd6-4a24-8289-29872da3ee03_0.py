from scipy.special import betaln

def compute_log_bayes_factor(alpha_0, beta_0, alpha_t, beta_t, alpha_c, beta_c, s_t, n_t, s_c, n_c):
    # Log probability of data under H0
    log_prob_data_H0 = betaln(alpha_0 + s_t + s_c, beta_0 + n_t + n_c - s_t - s_c) - betaln(alpha_0, beta_0)
    
    # Log probability of data under H1 for treatment group
    log_prob_data_H1_treatment = betaln(alpha_t + s_t, beta_t + n_t - s_t) - betaln(alpha_t, beta_t)
    
    # Log probability of data under H1 for control group
    log_prob_data_H1_control = betaln(alpha_c + s_c, beta_c + n_c - s_c) - betaln(alpha_c, beta_c)
    
    # Log joint probability of data under H1
    log_prob_data_H1 = log_prob_data_H1_treatment + log_prob_data_H1_control
    
    # Compute Log Bayes Factor
    log_BF_10 = log_prob_data_H1 - log_prob_data_H0
    
    return log_BF_10

# Example usage remains the same:
# ...

log_BF_10 = compute_log_bayes_factor(alpha_0, beta_0, alpha_t, beta_t, alpha_c, beta_c, s_t, n_t, s_c, n_c)
print(f"Log Bayes Factor (log_BF_10) = {log_BF_10}")
