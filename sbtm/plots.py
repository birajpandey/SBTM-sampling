import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import jax.numpy as jnp

def plot_distributions(initial_particles, transported_particles, density_params):
    fig = plt.figure(figsize=(10, 6))

    # Plot histogram of initial particles
    plt.hist(initial_particles, bins=30, density=True, alpha=0.4, color='b',
             histtype='bar', label='Initial Particles')

    # Plot histogram of transported particles
    plt.hist(transported_particles, bins=30, density=True, alpha=0.4,
             color='g', histtype='bar', label='Transported Particles')

    # Plot the target density function
    mean = density_params['mean'][0]
    std_dev = np.sqrt(density_params['covariance'][0, 0])
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    plt.plot(x, y, 'r-', lw=2, label='Target Distribution')

    # plt.title('Initial and Final Distributions of Particles')
    plt.xlabel('Particle Value', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return fig