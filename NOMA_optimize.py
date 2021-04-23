### optimize multiplicative NOMA constellation points
# Start: superimpose QAM constellations


import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

N=[4,8] # number of constellation points for each sender
iterations= 150


S1 = np.array([1,-1,1j,-1j])*np.exp(1j*np.pi/4)
#S1=rng.random((N[0],1)) + 1j*rng.random((N[0],1))
#S1= np.array([1+1j,1-1j,1+3j,1-3j,-1+1j,-1-1j,-1+3j,-1-3j,3+1j,3-1j,3+3j,3-3j,-3+1j,-3-1j,-3+3j,-3-3j])
S1=S1/np.max(np.abs(S1))

S2 = 1+rng.random((N[1],1)) + 1j*rng.random((N[1],1))
S2 = S2/np.max(S2)

C = np.transpose(S1)*S2

plt.scatter(np.real(C), np.imag(C))


#optimize S2:
S2n=S2
S1n=S1
S2prog=[]

for iter in range(iterations):
    """ for sym1 in range(N[0]):
        #print(C-S1[0]*S2n[sym])
        #print(sym)
        s2=rng.integers(0,N[1]) # choose random point from first constellation
        diff=(C-S1[sym1]*S2n[s2])
        points=np.nonzero(diff)

        # Symbol von dem nächsten Punkt wegschieben
        difn=np.array(diff[points]).flatten()
        mini=np.argmin(np.abs(difn))
        #print(difn[mini])
        d = difn[mini]/(np.abs(difn[mini])+1)/(iter+1)/S2n[s2]

        S1n[sym1]=S1n[sym1]-d
        S1n=S1n/np.max(np.abs(S1n))
        C=np.transpose(S1n)*S2n """

    for sym in range(N[1]):
        #print(C-S1[0]*S2n[sym])
        #print(sym)
        s1=rng.integers(0,4) # choose random point from first constellation
        diff=(C-S1n[s1]*S2n[sym])
        points=np.nonzero(diff)

        # Symbol von dem nächsten Punkt wegschieben
        difn=np.array(diff[points]).flatten()
        mini=np.argmin(np.abs(difn))
        #print(difn[mini])
        d = difn[mini]/(np.abs(difn[mini])+1)**5/(iter+1)/S1n[s1]

        S2n[sym]=S2n[sym]-d
        S2n=S2n/np.max(np.abs(S2n))
        C=np.transpose(S1n)*S2n
    S2prog.append(S2n)

plt.figure("optimized constellation after "+str(iter)+" iterations")
plt.scatter(np.real(C),np.imag(C))
print("Max Amplitude: " +str(np.max(np.abs(C))))

plt.figure("optimized S2")
for iter in range(iterations):
    if iter==0 or iter==iterations-1:
        plt.scatter(np.real(S2prog[iter]),np.imag(S2prog[iter]),label=iter)
    else:
        if iterations>200:
            pass
        else:
            plt.scatter(np.real(S2prog[iter]),np.imag(S2prog[iter]),color='k')
plt.legend()
print("S1= " + str(S1n))
print("S2= " + str(S2n))

plt.show()