### optimize multiplicative NOMA constellation points
# Start: superimpose QAM constellations


import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

N=[4,4,4] # number of constellation points for each sender
iterations= 200


S1 = np.array([1,-1,1j,-1j])*np.exp(1j*np.pi/4)
#S1 = np.exp(1j*np.pi/3*np.arange(6))
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

#optimize S3:
S3 = 1+0.2*rng.random((N[2],1)) + 0.2j*rng.random((N[2],1))
S3 = S3/np.max(S3)

S3n=S3
S3prog=[]

for iter in range(16*iterations):
    for sym in range(N[2]):
        #print(C-S1[0]*S2n[sym])
        #print(sym)
        s1=rng.integers(0,4,2) # choose random point from first constellation
        diff=(C-S1[s1[0]]*S2n[s1[1]]*S3n[sym])
        points=np.nonzero(diff)

        # Symbol von dem nächsten Punkt wegschieben
        difn=np.array(diff[points]).flatten()
        mini=np.argmin(np.abs(difn))
        #print(difn[mini])
        d = difn[mini]/(np.abs(difn[mini])+1)**5/(iter+1)/(S1n[s1[0]]*S2n[s1[1]])
        S3n[sym]=S3n[sym]-d
        S3n=S3n/np.max(np.abs(S3n))
        C=[np.transpose(S1)*S2n*S3n[0], np.transpose(S1)*S2n*S3n[1],np.transpose(S1)*S2n*S3n[2],np.transpose(S1)*S2n*S3n[3]]
    S3prog.append(S3n)

print("S3= " + str(S3n))

plt.figure("optimized constellation S3 after "+str(iter)+" iterations")
plt.scatter(np.real(C),np.imag(C))
print("Max Amplitude: " +str(np.max(np.abs(C))))

plt.figure("optimized S3")
for iter in range(16*
iterations):
    if iter==0 or iter==iterations-1:
        plt.scatter(np.real(S3prog[iter]),np.imag(S3prog[iter]),label=iter)
    else:
        if iterations>200:
            pass
        else:
            plt.scatter(np.real(S3prog[iter]),np.imag(S3prog[iter]),color='k')
plt.legend()

plt.show()