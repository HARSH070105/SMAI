import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.animation as animation
import os

os.makedirs("plots", exist_ok=True)

X = np.array([1, 2, 5, 8, 9])
mu = np.array([2.0, 4.0])
sigma = np.array([1.0, 1.0])
pi = np.array([0.5, 0.5])
x = np.linspace(0, 10, 500)
mu_hist, sigma_hist, pi_hist = [], [], []
log_likelihood_hist = []

def calculate_log_likelihood(X, mu, sigma, pi):
    return np.sum(np.log(
        pi[0]*norm.pdf(X, mu[0], sigma[0]) +
        pi[1]*norm.pdf(X, mu[1], sigma[1])
    ))

def step(mu, sigma, pi):
    resp = np.zeros((len(X), 2))
    for k in range(2):
        resp[:, k] = pi[k] * norm.pdf(X, mu[k], sigma[k])
    resp = resp / resp.sum(axis=1, keepdims=True)

    Nk = resp.sum(axis=0)
    mu_new = (resp.T @ X) / Nk
    sigma_new = np.sqrt(((resp * (X[:, None] - mu_new)**2).sum(axis=0)) / Nk)
    pi_new = Nk / len(X)

    return mu_new, sigma_new, pi_new

mu_hist.append(mu.copy())
sigma_hist.append(sigma.copy())
pi_hist.append(pi.copy())
log_likelihood_hist.append(calculate_log_likelihood(X, mu, sigma, pi))

for _ in range(20):
    mu, sigma, pi = step(mu, sigma, pi)
    mu_hist.append(mu.copy())
    sigma_hist.append(sigma.copy())
    pi_hist.append(pi.copy())
    log_likelihood_hist.append(calculate_log_likelihood(X, mu, sigma, pi))

fig_anim, ax_anim = plt.subplots()

def update(frame):
    ax_anim.clear()

    m = mu_hist[frame]
    s = sigma_hist[frame]
    p = pi_hist[frame]

    ax_anim.scatter(X, np.zeros_like(X), color='black', zorder=5)

    mixture = p[0]*norm.pdf(x, m[0], s[0]) + p[1]*norm.pdf(x, m[1], s[1])
    ax_anim.plot(x, mixture, linewidth=2, label='Mixture')

    ax_anim.plot(x, p[0]*norm.pdf(x, m[0], s[0]), '--', label=f'Comp 1 (μ={m[0]:.2f})')
    ax_anim.plot(x, p[1]*norm.pdf(x, m[1], s[1]), '--', label=f'Comp 2 (μ={m[1]:.2f})')

    ax_anim.set_xlim(0, 10)
    ax_anim.set_ylim(0, 0.5)
    ax_anim.set_title(f"EM Iteration {frame}")
    ax_anim.legend()

ani = animation.FuncAnimation(fig_anim, update, frames=len(mu_hist), interval=800)

plt.show()

fig1, axes = plt.subplots(2, 1, figsize=(10, 6))

m_init, s_init, p_init = mu_hist[0], sigma_hist[0], pi_hist[0]
axes[0].scatter(X, np.zeros_like(X), color='black', s=80)
axes[0].plot(x, p_init[0]*norm.pdf(x, m_init[0], s_init[0]), '--')
axes[0].plot(x, p_init[1]*norm.pdf(x, m_init[1], s_init[1]), '--')
axes[0].plot(x, p_init[0]*norm.pdf(x, m_init[0], s_init[0]) + 
                p_init[1]*norm.pdf(x, m_init[1], s_init[1]), color='purple')
axes[0].set_title("Initial GMM")

m_final, s_final, p_final = mu_hist[-1], sigma_hist[-1], pi_hist[-1]
axes[1].scatter(X, np.zeros_like(X), color='black', s=80)
axes[1].plot(x, p_final[0]*norm.pdf(x, m_final[0], s_final[0]), '--')
axes[1].plot(x, p_final[1]*norm.pdf(x, m_final[1], s_final[1]), '--')
axes[1].plot(x, p_final[0]*norm.pdf(x, m_final[0], s_final[0]) + 
                p_final[1]*norm.pdf(x, m_final[1], s_final[1]), color='purple')
axes[1].set_title("Final GMM")

plt.tight_layout()
plt.savefig("plots/gmm_initial_vs_final.png", dpi=150)
plt.show()

fig2, axes = plt.subplots(2, 1, figsize=(10, 6))
iterations = np.arange(len(log_likelihood_hist))

axes[0].plot(iterations, log_likelihood_hist, 'o-')
axes[0].set_title("Log-Likelihood Convergence")

axes[1].plot(iterations, [m[0] for m in mu_hist], label='μ₁', marker='o')
axes[1].plot(iterations, [m[1] for m in mu_hist], label='μ₂', marker='s')
axes[1].set_title("Means Convergence")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/gmm_convergence_metrics.png", dpi=150)
plt.show()