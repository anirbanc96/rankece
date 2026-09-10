import numpy as np
import matplotlib.pyplot as plt

from utils import phi_func, prob_func

m = 25
A = 0.2

x = np.linspace(0, 1, 50000)
z = A * (1-x) + (1-A) * x

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

axes[0].plot(z, phi_func(z, m, A))
axes[0].set_xlabel("z")
axes[0].set_ylabel("r(z)")
axes[0].set_title(f"Residual Function")

axes[1].plot(x, prob_func(x, m, A))
axes[1].plot(x, x, linestyle="--", color="gray")
axes[1].set_xlabel("x")
axes[1].set_ylabel("E[Y|X=x]")
axes[1].set_title(f"Conditional Probability")

plt.tight_layout()
plt.savefig("r_and_prob_func.pdf", dpi=1200)
plt.show()
