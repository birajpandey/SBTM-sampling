import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import jax.numpy as jnp
import jax.random as jrandom

from sbtm import density


# Define 1D reference
t = 0.1
cov = 1 - jnp.exp(-2 * t)
ref_params = {'mean': jnp.array([0]), 'variance': jnp.array([cov])}
ref_density_obj = density.Density(density.gaussian_pdf, ref_params)

# Define 1D gaussian target
target_params = {'mean': jnp.array([0]), 'variance': jnp.array([1.])}
target_density_obj = density.Density(density.gaussian_pdf, target_params)

# Initialize NN that learns the score

# Initialize the transport model

# Learn score of the initial distribution exactly

# for i in range(n_steps):

    # minimize score matching loss using transported particles on K steps

    # Transport particles based on Fokker Planck equation
    # x += dt * (target_score(x) - score_model(x))