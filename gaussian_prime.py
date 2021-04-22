### Gaussian Prime Decoding

import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

# 3 constellations taken from the set of Gaussian primes
G1=3*np.array([1+1j,1-1j,-1+1j,-1-1j])
G2=1/3*np.array([1+1j,1+2j, 2+1j,3])
G3=np.array([1+1j,3+2j,2+3j,1+4j])


num = 10000 # number of symbols
n1  = 0.2   # noise variance N_0/2
n2 = 0.3    # noise variance N_0/2

# signal creation: 
s1=rng.choice(G1,num)

# noise:
s1n = s1+ rng.normal(0,n1,num) + 1j*rng.normal(0,n1,num)

# signal 2
s2x = rng.choice(G2,num)
s2 = s1n*s2x

s2n = s2+ rng.normal(0,n2,num) + 1j*rng.normal(0,n2,num)

# plotting
plt.figure('Signal after Channel')
plt.scatter(np.real(s2n),np.imag(s2n))


# decode 1:
D2=np.zeros(num,complex)
v2=np.zeros(np.size(G2),complex)
error=0
# Teilen durch die zugeordneten Primzahlen; Vergleich mit nächst gerundetem Wert ---->> FIXME
for sig in range(num):
    for sym2 in range(np.size(G2)):
        v2[sym2]=np.abs(np.round(s2n[sig]/G2[sym2])-s2n[sig]/G2[sym2])
    D2[sig]=G2[np.argmin(v2)]
    if D2[sig]!=s2x[sig]:
        error=error+1

print("Errors for G2: "+str(error))

#plt.figure('SIC: nach demod von G2')
#plt.scatter(np.real(s2n/D2),np.imag(s2n/D2))
sd=s2n/D2

v1=np.zeros(np.size(G1),complex)
D1=np.zeros(num,complex)
error=0
# Vergleich mit G1 (Primzahl funktioniert hier nicht, weil streng genommen 1+j=j*(1+j) -> Diese Elemente repräsentieren die gleiche Primzahl)
for sig in range(num):
    for sym1 in range(np.size(G1)):
        v1[sym1]=np.abs(sd[sig]-G1[sym1])
    D1[sig]=G1[np.argmin(v1)]
    if D1[sig]!=s1[sig]:
        error=error+1

print("Errors for G1: "+str(error))
## not noise resilient, rounding to values chooses points that are not points of the signal constellation



error=[0,0]
D1=np.zeros(num,complex)
D2=np.zeros(num,complex)
v=np.zeros([np.size(G1),np.size(G2)],complex)

# decode 2: Joint Decoding
for sig in range(num):
    for sym2 in range(np.size(G2)):
        for sym1 in range(np.size(G1)):
            v[sym1,sym2]=np.abs(s2n[sig]/G2[sym2]-G1[sym1])
    D2[sig]=G2[np.mod(np.argmin(v),np.size(G1))]
    D1[sig]=G1[np.int(np.argmin(v)/np.size(G2))]
    if D1[sig]!=s1[sig]:
        error[0]=error[0]+1
    if D2[sig]!=s2x[sig]:
        error[1]=error[1]+1

print("Errors for G1: "+str(error[0]))
print("Errors for G2: "+str(error[1]))


plt.show()

##### The proposed Joint decoding performs A LOT better