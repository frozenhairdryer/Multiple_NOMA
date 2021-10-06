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

learnrate=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#learnrate=[0.0001,0.0003,0.0006]
runs=10
sum_sers=np.ones([len(learnrate),runs])
gmi_nn =np.zeros([len(learnrate),runs])
gmi_none =np.zeros([len(learnrate),runs])
gmi_div =np.zeros([len(learnrate),runs])
#gmi_all =np.zeros([len(learnrate),runs])
for lr in range(len(learnrate)):
    for num in range(runs):
        canc_method,enc_best,dec_best,canc_best, smi, validation_SERs,gmi_exact, snr, const =Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.18,0.18]),train_params=[50,300,learnrate[lr]],canc_method='nn', modradius=torch.tensor([1,1/2*np.sqrt(2)]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_nn[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        #sum_sers[lr,num]=sum_SERs[min_SER_iter]
        #gmi_nn[lr,num]=max(smi.detach().cpu().numpy())

plt.figure("GMI sweep NN",figsize=(2,2.5))
for num in range(runs):
    plt.scatter(learnrate,gmi_nn[:,num],alpha=0.5)
plt.scatter(learnrate,np.max(gmi_nn, axis=1))
plt.xlabel('learn rate')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_lrsweep_nn.pgf")

for lr in range(len(learnrate)):
    for num in range(runs):
        canc_method,enc_best,dec_best, smi, validation_SERs,gmi_exact=Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.18,0.18]),train_params=[50,300,learnrate[lr]],canc_method='none', modradius=torch.tensor([1,1/2*np.sqrt(2)]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_none[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        #sum_sers[lr,num]=sum_SERs[min_SER_iter]
        #gmi_none[lr,num]=max(smi.detach().cpu().numpy())

plt.figure("GMI sweep none",figsize=(2,2.5))
for num in range(runs):
    plt.scatter(learnrate,gmi_none[:,num],alpha=0.5)
plt.scatter(learnrate,np.max(gmi_none, axis=1))
plt.xlabel('learn rate')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_lrsweep_none.pgf")

for lr in range(len(learnrate)):
    for num in range(runs):
        canc_method,enc_best,dec_best, smi, validation_SERs,gmi_exact=Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.18,0.18]),train_params=[50,300,learnrate[lr]],canc_method='div', modradius=torch.tensor([1,1/2*np.sqrt(2)]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_div[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        #sum_sers[lr,num]=sum_SERs[min_SER_iter]
        #gmi_div[lr,num]=max(smi.detach().cpu().numpy())

plt.figure("GMI sweep div",figsize=(2,2.5))
for num in range(runs):
    plt.scatter(learnrate,gmi_div[:,num],alpha=0.5)
plt.scatter(learnrate,np.max(gmi_div, axis=1))
plt.xlabel('learn rate')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_lrsweep_div.pgf")
#plt.show()

with open('lrsweep.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([gmi_none,gmi_div,gmi_nn], f)