import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def pulsesinc(k,n):
    f = np.linspace(-1,1,201)
    p = np.zeros(len(f))
    pm = np.zeros(len(f))

    for kx in k:
        for nx in n:
            # factor 1/T**2* left out for normalization
            p += ((1-np.abs(f))**2)*((np.sinc((kx-nx)*(1-np.abs(f))))**2)/len(k)
   
    pm = (1-np.abs(f))**2
    #p = p/p[100]
    #pm = pm/pm[100]
    
    f = np.linspace(-3,3,601)
    p=np.pad(p,(200,200))
    pm=np.pad(pm,(200,200))
    return p,pm,f

def pulsegauss(k,n,a):
    f = np.linspace(-3,3,601)
    p = np.zeros(len(f))
    pm = np.pi/a*np.exp(-2/a*(np.pi*f)**2)
    for kx in k:
        for nx in n:
            p += np.pi/a*np.exp(-2/a*(np.pi*f)**2)*np.exp(-0.5*a*(kx-nx)**2)/len(k)
    return p,pm,f

def pulsrect(k,n):
    f = np.linspace(-3,3,601)
    p = np.zeros(len(f))
    pm = np.sinc(f)**2
    p = pm
    return p,pm,f



matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size' : 10,
})
    
cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
new_color_list = np.array([[t/2 + 0.49 for t in color_list[k]] for k in range(len(color_list))])

k = np.linspace(-100,100,201)
n = np.linspace(-100,100,201)


plt.figure(figsize=(5,3))

psdx,psdmx,f = pulsesinc(k,n)
plt.plot(f,psdx,'--', color= color_list[0],label="signal with ISI")
plt.plot(f,psdmx,'r',linewidth=2, label="ideal ISI-free signal")
plt.xlabel('normalized Frequency $f$')
plt.ylabel('PSD')
plt.grid('--')
plt.tight_layout()
plt.legend(loc=1)
plt.savefig("sincpulsefreq.pdf")


plt.figure(figsize=(5,3))
B = 1
a = 5
psdx,psdmx,f = pulsegauss(k,n,a)
#psd,psdm,f = pulsegauss(k,n,20)
plt.plot(f,psdx/max(psdmx),'--', color= color_list[0],label="signal with ISI")
plt.plot(f,psdmx/max(psdmx),'r',linewidth=2, label="ideal ISI-free signal")

plt.xlabel('normalized Frequency $f$')
plt.ylabel('PSD')
plt.grid('--')
plt.tight_layout()
plt.legend(loc=1)
plt.savefig("gausspulsefreq.pdf")

plt.figure(figsize=(5,3))

psdx,psdmx,f = pulsrect(k,n)
plt.plot(f,psdx,'--', color= color_list[0],label="signal with ISI")
plt.plot(f,psdmx,'r',linewidth=2, label="ideal ISI-free signal")
plt.xlabel('normalized Frequency $f$')
plt.ylabel('PSD')
plt.grid('--')
plt.tight_layout()
plt.legend(loc=1)
plt.savefig("rectpulsefreq.pdf")
