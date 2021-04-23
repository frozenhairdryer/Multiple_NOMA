### optimize multiplicative NOMA constellation points
# Start: superimpose QAM constellations


import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

N=[4,8] # number of constellation points for each sender
iterations= 100


S1 = np.array([1,-1,1j,-1j])*np.exp(1j*np.pi/4)

S2 = 1+rng.random((N[1],1)) + 1j*rng.random((N[1],1))
S2 = S2/np.max(S2)

C = S1*S2

plt.scatter(np.real(C), np.imag(C))


#optimize S2:
S2n=S2
S2prog=[]

for iter in range(iterations):
    for sym in range(N[1]):
        #print(C-S1[0]*S2n[sym])
        #print(sym)
        s1=rng.integers(0,4) # choose random point from first constellation
        diff=(C-S1[s1]*S2n[sym])
        points=np.nonzero(diff)

        # Symbol von dem nächsten Punkt wegschieben
        difn=np.array(diff[points]).flatten()
        mini=np.argmin(np.abs(difn))
        #print(difn[mini])
        d = difn[mini]/(np.abs(difn[mini])+1)/(iter+1)/S1[s1]

        S2n[sym]=S2n[sym]-d
        S2n=S2n/np.max(np.abs(S2n))
        C=S1*S2n
    S2prog.append(S2n)

plt.figure("optimized constellation after "+str(iter)+" iterations")
plt.scatter(np.real(C),np.imag(C))
print("Max Amplitude: " +str(np.max(np.abs(C))))

plt.figure("optimized S2")
for iter in range(iterations):
    if iter==0 or iter==iterations-1:
        plt.scatter(np.real(S2prog[iter]),np.imag(S2prog[iter]),label=iter)
    else:
        plt.scatter(np.real(S2prog[iter]),np.imag(S2prog[iter]),color='k')
plt.legend()
print(S2n)

plt.show()