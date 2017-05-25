import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0,50, 200)

delta_0 = 10

func = lambda l: (lambda t: delta_0 * (1 + delta_0*l*t)**(-1))
handles = []

ax3 = plt.plot(xs, list(map(func(0.01), xs)), '-.', color='k', linewidth=0.8, label='$\gamma=0.01$')
handles.extend(ax3)

ax2 = plt.plot(xs, list(map(func(0.1), xs)), '--', color='k', linewidth=0.8, label='$\gamma=0.1$')
handles.extend(ax2)

ax1 = plt.plot(xs, list(map(func(1.0), xs)), '-', color='k', linewidth=0.8, label='$\gamma=1$')
handles.extend(ax1)




plt.legend(handles=handles)

plt.savefig('report/learning-rates.pdf')
