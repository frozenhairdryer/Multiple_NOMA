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
    #p = p/np.sum(p)
    #pm = pm/pm[100]
    
    f = np.linspace(-4,4,801)
    p=np.pad(p,(300,300))
    pm=np.pad(pm,(300,300))
    p = 10*np.log10(p/np.max(p)+0.00000001)
    return p,pm,f

def pulsegauss(k,n,a):
    f = np.linspace(-4,4,801)
    p = np.zeros(len(f))
    #pm = np.pi/(2*a)*np.exp(-1/a*(np.pi*f)**2)
    pm = np.exp(-2/a*(np.pi*f)**2)
    for kx in k:
        for nx in n:
            p += np.pi/(2*a)*np.exp(-1/a*(np.pi*f)**2)*np.exp(-a*(kx-nx)**2)/len(k)
    #p = p/np.sum(p)
    p = 10*np.log10(p/np.max(p)+0.00000001)
    pm = 10*np.log10(pm/np.max(pm)+0.00000001)
    return p,pm,f

def pulsrect(k,n):
    f = np.linspace(-4,4,801)
    #p = np.zeros(len(f))
    p = (np.sinc(f))**2
    #p = pm
    #p = p/np.sum(p)
    pm=p
    #p /= np.sum(p)
    pm = 10*np.log10(pm/np.max(pm)+0.00000001)
    p = 10*np.log10(p/np.max(p)+0.00000001)
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

psdx_sinc,psdmx,f = pulsesinc(k,n)
psd_add = np.zeros(len(f))-60
for fi in range(len(f)):
    if f[fi]>=-0.5 and f[fi]<=0.5:
        psd_add[fi] = 0

plt.plot(f,psdx_sinc, color= color_list[0], label=r'Multipl. signal')
plt.plot(f,psd_add, color = color_list[4], label=r'Added signal')
#plt.plot(f,psdmx,'r',linewidth=2, label=r"$n=k$")
plt.xlabel(r'normalized Frequency $fT$')
plt.ylabel(r'PSD [dB]')
plt.grid('--')
plt.ylim(-50,10)
plt.tight_layout()
plt.legend(loc=1)
plt.savefig("sincpulsefreq.pdf")


plt.figure(figsize=(5,3))
a = 2.5
psdx_gauss,psdmx,f = pulsegauss(k,n,a)

#psd,psdm,f = pulsegauss(k,n,20)
plt.plot(f,psdx_gauss, color= color_list[0], label=r'Multipl. signal')
plt.plot(f,psdmx, color= color_list[4], label=r'Added signal')
plt.xlabel(r'normalized Frequency $fT$')
plt.ylabel(r'PSD [dB]')
plt.grid('--')
plt.ylim(-50,10)
plt.tight_layout()
plt.legend(loc=1)
plt.savefig("gausspulsefreq.pdf")

plt.figure(figsize=(5,3))

psdx_rect,psdmx,f = pulsrect(k,n)
plt.plot(f,psdx_rect, color= color_list[0], label=r'Multipl. signal')
plt.plot(f,psdmx, color= color_list[4], label=r'Added signal')
plt.xlabel(r'normalized Frequency $fT$')
plt.ylabel(r'PSD [dB]')
plt.grid('--')
plt.tight_layout()
plt.ylim(-50,10)
plt.legend(loc=1)
plt.savefig("rectpulsefreq.pdf")

plt.figure("PSDs",figsize=(3,3))
plt.plot(f,psdx_rect, color= color_list[0],label=r'Rect')
plt.plot(f,psdx_sinc, color= color_list[2],label=r'Sinc')
plt.plot(f,psdx_gauss, color= color_list[4], label=r'Gauss')
plt.xlabel(r'normalized Frequency $fT$')
plt.ylabel(r'PSD [dB]')
plt.grid('--')
plt.legend(loc=1)
plt.ylim(-50,10)
plt.tight_layout()
plt.savefig("PSDs.pdf")

