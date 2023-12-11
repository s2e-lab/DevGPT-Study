import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = 1/np.sin(x)

plt.figure(figsize=(8,6))
plt.plot(x, y)
plt.ylim(-10, 10) # limit y-axis to avoid very high values at discontinuities
plt.title(r'$y=\frac{1}{\sin(x)}$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
