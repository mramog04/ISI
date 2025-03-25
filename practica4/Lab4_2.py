import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 1000)
y = np.linspace(0, 20, 1000)


fig, ax = plt.subplots(3,1, figsize=(10, 10))

ax[0].plot(x, np.cos(x), '-r', label='Cosine')
ax[0].plot(x, np.cos(x + np.pi/4),'--b', label='Cosine')
ax[0].grid(axis='y')
ax[0].axis([0, 20, -1, 1])
ax[0].set_xlabel('x axis')
ax[0].set_ylabel('y axis')


ax[1].plot(x, np.sin(x), '-r', label='Sine')
ax[1].plot(x, np.sin(x + np.pi/4), '--b', label='Sine')
ax[1].grid(axis='y')
ax[1].axis([0, 20, -1, 1])
ax[1].set_xlabel('x axis')
ax[1].set_ylabel('y axis')



ax[2].plot(x, np.tan(x), '-r', label='Tangent')
ax[2].plot(x, np.tan(x + np.pi/4),'--b', label='Tangent')
ax[2].grid(axis='y')
ax[2].axis([0, 20, -10, 10])
ax[2].set_xlabel('x axis')
ax[2].set_ylabel('y axis')

plt.show()