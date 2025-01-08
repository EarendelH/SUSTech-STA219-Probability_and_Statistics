import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def target_distribution(x):
    return 0.6 * np.exp(-(x + 5)**2 / 2) + 0.4 * np.exp(-(x - 1)**2 / 0.5)


class MixtureNormal:
    def __init__(self, weights, means, stds):
        self.weights = weights
        self.means = means
        self.stds = stds

    def pdf(self, x):
        return sum(w * norm.pdf(x, m, s) for w, m, s in zip(self.weights, self.means, self.stds))

    def rvs(self, size=1):
        component = np.random.choice(len(self.weights), size=size, p=self.weights)
        return np.random.normal(self.means[component[0]], self.stds[component[0]], size=size)

weights = [0.65, 0.35]
means = [-5, 1]
stds = [1,0.6]
proposal_distribution = MixtureNormal(weights, means, stds)

c = 2.5

x = np.linspace(-10, 10, 1000)
plt.plot(x, target_distribution(x), label='Target Distribution f*(x)')
plt.plot(x, c * proposal_distribution.pdf(x), label='Scaled Proposal Distribution c*g(x)')
plt.legend()
plt.title('Target and Proposal Distributions')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()


def rejection_sampling(target, proposal, c, num_samples):
    samples = []
    count = 0
    
    while len(samples) < num_samples:
        x = proposal.rvs()
        u = np.random.uniform(0, c * proposal.pdf(x))
        if u < target(x):
            samples.append(x)
        count += 1
    return np.array(samples), count


num_samples = 50000

samples, total_count = rejection_sampling(target_distribution, proposal_distribution, c, num_samples)

acceptance_ratio = num_samples / total_count
print(f'Acceptance Ratio: {acceptance_ratio:.4f}')

plt.hist(samples, bins=100, density=True, alpha=0.6, color='g', edgecolor='black', label='Sampled Data')

normalized_target = target_distribution(x) / np.trapz(target_distribution(x), x)
plt.plot(x, normalized_target, 'r', linestyle='--', label='Normalized Target Distribution f(x)')

plt.legend()
plt.suptitle('Rejection Sampling Results')
plt.title('Acceptance Ratio: {:.4f}'.format(acceptance_ratio))
plt.xlabel('x')
plt.ylabel('Density')
plt.show()