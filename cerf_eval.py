import numpy as np
from scipy import special
import matplotlib.pyplot as plt

z =(1+1j)*np.linspace(-10,10,200,dtype=complex)

s = special.erf(z)-special.erf(z-(1+1j))


plt.subplot(211)
plt.plot(np.real(z),np.real(s))

plt.subplot(212)
plt.plot(np.real(z),np.imag(s))
plt.savefig("erf_eval.pdf")