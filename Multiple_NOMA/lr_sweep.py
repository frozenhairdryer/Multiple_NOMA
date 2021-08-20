from training_routine import *
import numpy as np

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
gmi_all =np.zeros([len(learnrate),runs])
for lr in range(len(learnrate)):
    for num in range(runs):
        canc_method,enc_best,dec_best,canc_best, smi, validation_SERs,gmi_exact=Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.09,0.09]),train_params=[50,300,learnrate[lr]],canc_method='nn', modradius=torch.tensor([1,1/3*np.sqrt(2)]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_all[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        sum_sers[lr,num]=sum_SERs[min_SER_iter]
        gmi_nn[lr,num]=max(smi.detach().cpu().numpy())

plt.figure("GMI sweep",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(learnrate,gmi_nn[:,num], c='blue',alpha=0.5)
plt.scatter(learnrate,np.max(gmi_nn, axis=1))
plt.xlabel('learn rate')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_lrsweep_nn_dp.pgf")

for lr in range(len(learnrate)):
    for num in range(runs):
        canc_method,enc_best,dec_best,canc_best, smi, validation_SERs,gmi_exact=Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.09,0.09]),train_params=[50,300,learnrate[lr]],canc_method='none', modradius=torch.tensor([1,1/3*np.sqrt(2)]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_all[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        sum_sers[lr,num]=sum_SERs[min_SER_iter]
        gmi_none[lr,num]=max(smi.detach().cpu().numpy())

plt.figure("GMI sweep",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(learnrate,gmi_none[:,num], c='blue',alpha=0.5)
plt.scatter(learnrate,np.max(gmi_none, axis=1))
plt.xlabel('learn rate')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_lrsweep_none_dp.pgf")

for lr in range(len(learnrate)):
    for num in range(runs):
        canc_method,enc_best,dec_best,canc_best, smi, validation_SERs,gmi_exact=Multipl_NOMA(M=torch.tensor([4,4]),sigma_n=torch.tensor([0.09,0.09]),train_params=[50,300,learnrate[lr]],canc_method='div', modradius=torch.tensor([1,1/3*np.sqrt(2)]), plotting=False)
        sum_SERs = np.sum(validation_SERs.detach().cpu().numpy(), axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs.detach().cpu().numpy(),axis=0))
        gmi_all[lr,num] = max(np.sum(gmi_exact.detach().cpu().numpy(), axis=1))
        #max_GMI = np.argmax(smi)
        sum_sers[lr,num]=sum_SERs[min_SER_iter]
        gmi_div[lr,num]=max(smi.detach().cpu().numpy())

plt.figure("GMI sweep",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(learnrate,gmi_div[:,num], c='blue',alpha=0.5)
plt.scatter(learnrate,np.max(gmi_div, axis=1))
plt.xlabel('learn rate')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()
plt.savefig("GMI_lrsweep_div_dp.pgf")
#plt.show()