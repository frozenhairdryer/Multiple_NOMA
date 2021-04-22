### optimize multiplicative NOMA constellation points
# Start: superimpose QAM constellations
# WIP

import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

N=[4,4] # number of constellation points for each sender

S1 = np.array([1,-1,1j,-1j])*np.exp(1j*np.pi/4)

S2 = 1+rng.random((N[1],1)) + 1j*rng.random((N[1],1))
S2 = S2/np.max(S2)

C = S1*S2

plt.scatter(np.real(C), np.imag(C))


#optimize S2:
S2n=S2
diff=[]

for iter in range(15):
    for sym in range(N[1]):
        for sym1 in range(N[0]):
            diff.append(np.delete(C,sym)-S1[sym1]*S2n[sym])
        difn=np.array(diff).flatten()
        #print(difn)
        #mini=np.argmin(np.abs(difn))
        d=0
        for abstand in range(np.size(difn)):
            if np.isnan(difn[abstand])==True or np.abs(difn[abstand])>1:
                pass
            else:
                d = d-difn[abstand]/(np.abs(difn[abstand])+1)/(iter+1) # funktioniert noch nicht ganz, FIXME: überlegen ob statisch - okay ist
        S2n[sym]=S2n[sym]+d
        C=S1*S2n
    S2n=S2n/np.max(np.abs(S2n))
    C=S1*S2n

plt.figure("optimized constellation after "+str(iter)+" iterations")
plt.scatter(np.real(C),np.imag(C))
print("Max Amplitude: " +str(np.max(np.abs(C))))

plt.figure("optimized S2")
plt.scatter(np.real(S2n),np.imag(S2n))
print(S2n)

plt.show()