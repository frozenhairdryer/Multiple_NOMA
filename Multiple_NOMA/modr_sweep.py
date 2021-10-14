from training_routine import *
import numpy as np
import pickle

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    #'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 8
})

cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors

length = 41
modr = torch.linspace(0,1,length)

runs=20
sum_sers=np.ones([length,runs])
gmi_nn =np.zeros([length,runs])
mod_calc = np.zeros([length,runs,2])
#gmi_all =np.zeros([length,runs])

gmi_nn,mod_calc, modr = pickle.load( open( "modrsweep_n.pkl", "rb" ) )

""" for lr in range(length):
    for num in range(runs):
        canc_method,enc_best,dec_best, mod, validation_SERs,gmi_exact, snr, const=Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.18,0.18]),train_params=[30,600,0.005],canc_method='div', modradius=torch.tensor([1,modr[lr]]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_nn[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        mod_calc[lr,num] = mod
        #sum_sers[lr,num]=sum_SERs[min_SER_iter]
        #gmi_nn[lr,num]=max(smi.detach().cpu().numpy()) """

#with open('modrsweep.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([gmi_nn,mod_calc, modr], f)

plt.figure("GMI modr sweep",figsize=(3,2.2))
for num in range(runs):
    plt.scatter(modr.detach().cpu().numpy()/np.sqrt(2),gmi_nn[:,num],color=color_list[0],alpha=0.5)
plt.plot(modr.detach().cpu().numpy()/np.sqrt(2),np.max(gmi_nn, axis=1), color=color_list[2],linewidth=1, label='Max')
plt.plot(modr.detach().cpu().numpy()/np.sqrt(2),np.mean(gmi_nn, axis=1), color=color_list[4],linewidth=1, label='Mean')
plt.xlabel(r'Modulation Radius $\frac{\alpha}{d_{min}}$')
plt.ylabel("GMI")
plt.grid()
plt.ylim(2,4)
plt.xlim(0,1/np.sqrt(2))
plt.legend(loc=4)
plt.tight_layout()
plt.savefig("GMI_modrsweep.pgf")

plt.figure("GMI modr sweep calc",figsize=(3,2))
for num in range(runs):
    plt.scatter(modr.detach().cpu().numpy(),mod_calc[:,num,1], color=color_list[0], alpha=0.5)
plt.plot(modr.detach().cpu().numpy(),np.mean(mod_calc[:,:,1], axis=1), color=color_list[4])
#plt.plot(modr.detach().cpu().numpy(),np.max(gmi_nn, axis=1))sc
plt.xlabel('Modulation Radius given')
plt.ylabel("Radius calculated")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_modrsweep_calc.pgf")

plt.figure("GMI vs modr calc",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(mod_calc[:,num,1],gmi_nn[:,num],color=color_list[0],alpha=0.5)
#plt.plot(modr.detach().cpu().numpy(),np.max(gmi_nn, axis=1), color=color_list[2], label='Max')
#plt.plot(modr.detach().cpu().numpy(),np.mean(gmi_nn, axis=1), color=color_list[4], label='Mean')
plt.xlabel('Radius calculated')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.legend(loc=4)
plt.savefig("GMI_vs_modr_calc.pgf")


