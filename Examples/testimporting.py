from matplotlib import pyplot as plt
import numpy as np

from quocslib.utils.testimportfunction import gaussian

xs = np.linspace(0,1,100)

plt.plot(xs, gaussian(xs,1,0.5,0))
plt.show()