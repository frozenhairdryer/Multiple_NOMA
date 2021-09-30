from training_routine import *
import numpy as np
import pickle

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors

length = 41
sigma_all = 0.3 #0.3
sigma = np.zeros((length,2))
for x in range(length):
    sigma[x,0]=np.sqrt((x+1)/41*sigma_all**2) # percentile growth of noise energy
    sigma[x,1]=np.sqrt(sigma_all**2-sigma[x,0]**2) # absolute noise energy stays constant


runs=10
sum_sers=np.ones([length,runs])
gmi =np.zeros([length,runs])
mod_calc = np.zeros([length,runs,2])
snr_all = np.zeros([length,runs])
const_form = np.zeros([length,runs]) # number of symbols with amplitudes lower than 0.9

#gmi_all =np.zeros([length,runs])
for x in range(length):
    for num in range(runs):
        canc_method,enc_best,dec_best, mod, validation_SERs,gmi_exact, snr,const =Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor(sigma[x]),train_params=[30,600,0.005],canc_method='div', modradius=torch.tensor([1,0.58]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi[x,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        max_GMI = np.argmax(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        mod_calc[x,num] = mod
        snr_all[x,num] = snr[max_GMI]
        for elem in const:
            if np.abs(elem)<0.9:
                const_form[x,num] += 1
        #sum_sers[lr,num]=sum_SERs[min_SER_iter]
        #gmi[lr,num]=max(smi.detach().cpu().numpy())


with open('noiseweight.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([sigma, gmi, snr_all, const_form], f)

plt.figure("GMI sigma weight",figsize=(3,2.5))
for num in range(runs):
    for x in range(length):
        plt.scatter(cp.asnumpy(sigma[x,0])**2,gmi[x,num],color=color_list[const_form[x,num]],alpha=0.5)
plt.plot(cp.asnumpy(sigma[:,0])**2,np.max(gmi, axis=1), color=color_list[2], label='Max')
plt.plot(cp.asnumpy(sigma[:,0])**2,np.mean(gmi, axis=1), color=color_list[4], label='Mean')
plt.xlabel(r'$\sigma_1^2$')
plt.ylabel("GMI (bit)")
plt.grid()
plt.tight_layout()
plt.legend(loc=4)
plt.savefig("GMI_noiseweight.pgf")

plt.figure("SNR sigma weight",figsize=(3,2.5))
for num in range(runs):
    for x in range(length):
        plt.scatter(cp.asnumpy(sigma[x,0])**2,snr_all[x,num],color=color_list[const_form[x,num]],alpha=0.5)
plt.plot(cp.asnumpy(sigma[:,0])**2,np.mean(snr_all[:,:], axis=1), color=color_list[4])
#plt.plot(modr.detach().cpu().numpy(),np.max(gmi, axis=1))
plt.xlabel(r'$\sigma_1^2$')
plt.ylabel("SNR (dB)")
plt.grid()
plt.tight_layout()
plt.savefig("SNR_noiseweight.pgf")

plt.figure("SNR vs GMI weight",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(snr_all[:,num], gmi[:,num], color=color_list[const_form[:,num]], alpha=0.5)
#plt.plot(cp.asnumpy(np.mean(snr_all[:,:], axis=1), color=color_list[4])
#plt.plot(modr.detach().cpu().numpy(),np.max(gmi, axis=1))
plt.ylabel(r'GMI$_{joint} (bit)$')
plt.xlabel("SNR (dB)")
plt.grid()
plt.tight_layout()
plt.savefig("SNR_vs_GMI_noiseweight.pgf")


