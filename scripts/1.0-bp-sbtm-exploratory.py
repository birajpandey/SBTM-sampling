import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sbtm import density, plots


# Define reference measure
t0 = 0.01
cov = 1 - jnp.exp(-2 * t0)
ref_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[cov]])}
ref_density_obj = density.Density(density.gaussian_pdf, ref_params)

# generate particles from reference measure
key = jrandom.PRNGKey(0)
particles = jrandom.multivariate_normal(key, ref_params['mean'],
                                        ref_params['covariance'],
                                        shape=(1000, ))

# Define target
target_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[1.]])}
target_density_obj = density.Density(density.gaussian_pdf, target_params)

# plot
x = np.linspace(-4, 4, 1000)
fig = plt.figure(figsize=(10, 6))
fig.suptitle('Analytical solution of Linear FPE',
             fontsize=25)
plt.plot(x, ref_density_obj(x), 'g-', lw=2, label=f'$f_0$')
plt.plot(x, target_density_obj(x), 'r-', lw=2, label='$f_\infty$')
# plot intermediate densities
for t in np.arange(t0, 1.1, 0.1):
    model_params = {'dimension': 1, 'time': t}
    model_density_obj = density.Density(density.linear_FPE_pdf, model_params)
    plt.plot(x, model_density_obj(x), '--', lw=1, label=f't={t:0.2f}')
    plt.legend()

plt.xlabel('Particle Value', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.show()


# Transport based on true solution
dt = 0.01
t_steps = np.arange(t0, 1.1 + dt, dt)
trajectory = np.zeros((len(particles), len(t_steps) + 1))
trajectory[:, 0] = particles.flatten()
for i, t in tqdm(enumerate(t_steps)):

    # use true solution
    model_params = {'dimension': 1, 'time': t}
    model_density_obj = density.Density(density.linear_FPE_pdf, model_params)

    # update the particles
    model_score = model_density_obj.score(particles)
    target_score = target_density_obj.score(particles)
    gradient = dt * (target_score - model_score)
    particles += (gradient)

    print(f'time={t}, '
          f'Target Score norm={np.linalg.norm(target_score)}, '
          f'Model Score norm={np.linalg.norm(model_score)},'
          f'Gradient Score norm={np.linalg.norm(gradient)},')

    # save
    trajectory[:, i + 1] = particles.flatten()


# plot
fig = plots.plot_distributions(trajectory[:, 0], trajectory[:, -1],
                               target_params)
fig.suptitle(r'Transport under true $\nabla \log f_t$', fontsize=25)
fig.show()










