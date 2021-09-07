from training_routine import *
import numpy as np
import pickle

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    #'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

#learnrate=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#learnrate=[0.0001,0.0003,0.0006]
length = 41
modr = torch.linspace(0,1,length)

runs=10
sum_sers=np.ones([length,runs])
gmi_nn =np.zeros([length,runs])
mod_calc = np.zeros([length,runs,2])
#gmi_all =np.zeros([length,runs])
for lr in range(length):
    for num in range(runs):
        canc_method,enc_best,dec_best, mod, validation_SERs,gmi_exact=Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.18,0.18]),train_params=[50,600,0.005],canc_method='div', modradius=torch.tensor([1,modr[lr]]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_nn[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        mod_calc[lr,num] = mod
        #sum_sers[lr,num]=sum_SERs[min_SER_iter]
        #gmi_nn[lr,num]=max(smi.detach().cpu().numpy())

plt.figure("GMI modr sweep",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(modr.detach().cpu().numpy(),gmi_nn[:,num],alpha=0.5)
plt.plot(modr.detach().cpu().numpy(),np.max(gmi_nn, axis=1))
plt.xlabel('Modulation Radius')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_modrsweep.pgf")

plt.figure("GMI modr sweep calc",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(modr.detach().cpu().numpy(),mod_calc[:,num,1], alpha=0.5)
#plt.plot(modr.detach().cpu().numpy(),np.max(gmi_nn, axis=1))
plt.xlabel('Modulation Radius given')
plt.ylabel("Modulation Radius calculated")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_modrsweep_calc.pgf")



with open('modrsweep.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([gmi_nn,mod_calc, modr], f)