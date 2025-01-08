import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, cauchy

p = 0.5
geom_data = np.random.geometric(p, 10000)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(geom_data, bins=range(1, max(geom_data)+1), density=True, alpha=0.6, color='g', edgecolor='blue')

x = np.arange(1, max(geom_data)+1)
pmf = geom.pmf(x, p)
plt.plot(x, pmf, 'r', marker='o', linestyle='--')

plt.title('Geometric Distribution (p=0.5)')
plt.xlabel('Value')
plt.ylabel('Frequency')

cauchy_data = np.random.standard_cauchy(10000)

plt.subplot(1, 2, 2)
plt.hist(cauchy_data, bins=1000, density=True, alpha=0.6, color='g', edgecolor='blue', range=(-25, 25))

x = np.linspace(-25, 25, 1000)
pdf = cauchy.pdf(x)
plt.plot(x, pdf, 'r', linestyle='--')

plt.title('Standard Cauchy Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()