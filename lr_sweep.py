from functions_serialform import *

learnrate=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#learnrate=[0.0001,0.0003,0.0006]
runs=10
sum_sers=np.ones([len(learnrate),runs])
gmi =np.zeros([len(learnrate),runs])
for lr in range(len(learnrate)):
    for num in range(runs):
        canc_method,enc_best,dec_best, smi, validation_SERs=Multipl_NOMA(M=[4,4],sigma_n=[0.08,0.08],train_params=[50,300,learnrate[lr]],canc_method='div', modradius=[1,1.5/3*np.sqrt(2)], plotting=False)
        sum_SERs = np.sum(validation_SERs, axis=0)/2
        min_SER_iter = np.argmin(np.sum(validation_SERs,axis=0))
        #max_GMI = np.argmax(smi)
        sum_sers[lr,num]=sum_SERs[min_SER_iter]
        gmi[lr,num]=max(smi)

plt.figure("GMI sweep",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(learnrate,gmi[:,num], c='blue',alpha=0.5)
plt.scatter(learnrate,np.max(gmi, axis=1))
plt.xlabel('learn rate')
plt.ylabel("GMI")
plt.grid()
plt.tight_layout()

plt.figure("SER sweep",figsize=(3,2.5))
for num in range(runs):
    plt.scatter(learnrate,sum_sers[:,num],c='blue',alpha=0.5)
plt.scatter(learnrate,np.max(sum_sers, axis=1))
plt.xlabel('learn rate')
plt.ylabel("summed SER")
plt.grid()
plt.tight_layout()

plt.show()