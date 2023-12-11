import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Gaussian distribution
mu = 5.0
sigma = 2.0
N = 100  # Number of samples
K = 10   # Number of bins

# Generate samples from a Gaussian distribution
samples = np.random.normal(mu, sigma, N)

# Step 1: Generate the histogram
hist, bins = np.histogram(samples, bins=K, density=False)

# Step 2: Bootstrap resampling
n_resamples = 1000
resampled_histograms = []
for _ in range(n_resamples):
    resample = np.random.choice(samples, size=N, replace=True)
    resampled_hist, _ = np.histogram(resample, bins=K, density=False)
    resampled_histograms.append(resampled_hist)

# Step 3: Calculate statistics
mean_counts = np.mean(resampled_histograms, axis=0)
std_dev_counts = np.std(resampled_histograms, axis=0)

# Step 4: Assign error bars
error_bars = std_dev_counts

# Step 5: Plot the histogram with error bars
plt.bar(bins[:-1], mean_counts, width=np.diff(bins), yerr=error_bars, align='edge')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.title('Histogram with Error Bars')
plt.show()
