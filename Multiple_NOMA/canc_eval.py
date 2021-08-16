import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from training_routine import *
import pickle


### parameters
runs = 50
num_epochs=60

sigma_n=torch.tensor([0.09,0.09])
M=torch.tensor([4,4])
alph=torch.tensor([1,1/3*np.sqrt(2)])
#alph=[1,1]

params=[runs,num_epochs,sigma_n,M,alph]
best_impl=[[0,0],0,0,0,0]

## NN canceller 
GMI_nncanc=[]
list_nncanc=[]
#compare_data.append([])
for item in range(runs):
    #compare_data[2].append(dir())
    #plt.close('all')
    _,en, dec,canc, gmi, ser, gmi_exact = Multipl_NOMA(M,sigma_n,train_params=[num_epochs,300,0.002],canc_method='nn', modradius=alph, plotting=False)
    g = gmi_exact.detach().cpu().numpy()
    GMI_nncanc.append(np.sum(g, axis=1))
    list_nncanc.append(ser)
    if max(np.sum(g, axis=1))>max(best_impl[0]):
        best_impl=[np.sum(g, axis=1),en, dec, canc, ser]
        best_achieved='nn'

## save all data if further processing is wanted
with open('Multiple_NOMA/cancel_eval_gmiexact_NN_bw.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([GMI_nncanc, best_impl, best_achieved, params], f)
#print("Best implementation achieved with "+best_achieved+" cancellation.")

## figures
cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
#new_color_list = [[t/4 + 0.75 for t in color_list[k]] for k in range(len(color_list))]


#print(np.shape(np.array(GMI_nocanc)))



for item in range(runs):
    if item==0:
        average_nncanc = 1/runs*GMI_nncanc[item]
    else:
        average_nncanc += 1/runs*GMI_nncanc[item]
    
#average_nocanc = np.mean(GMI_nocanc, axis=1)
#average_divcanc = np.mean(GMI_divcanc, axis=0)
#average_nncanc = np.mean(GMI_nncanc)

#var_nocanc=np.var(GMI_nocanc, axis=1)
#var_divcanc=np.var(GMI_divcanc, axis=0)
#var_nncanc=np.var(GMI_nncanc)
#print(list_nocanc)
#print(np.min(list_nocanc, axis=1))

# variance:

# for item in range(runs):
#     if item==0:
#         # var_nocanc0=(list_nocanc[item][:][0]-average_nocanc0)**2/runs
#         # var_nocanc1=(list_nocanc[item][:][1]-average_nocanc1)**2/runs
#         # var_dcanc0=(list_divcanc[item][:][0]-average_dcanc0)**2/runs
#         # var_dcanc1=(list_divcanc[item][:][1]-average_dcanc1)**2/runs
#         # var_nncanc0=(list_nncanc[item][:][0]-average_nocanc0)**2/runs
#         # var_nncanc1=(list_nncanc[item][:][1]-average_nocanc1)**2/runs
#         var_nocanc=(GMI_nocanc[item]-average_nocanc)**2/runs
#         var_divcanc=(GMI_divcanc[item]-average_divcanc)**2/runs
#         var_nncanc=(GMI_nncanc[item]-average_nncanc)**2/runs
#     else:
#         # var_nocanc0+=(list_nocanc[item][:][0]-average_nocanc0)**2/runs
#         # var_nocanc1+=(list_nocanc[item][:][1]-average_nocanc1)**2/runs
#         # var_dcanc0+=(list_divcanc[item][:][0]-average_dcanc0)**2/runs
#         # var_dcanc1+=(list_divcanc[item][:][1]-average_dcanc1)**2/runs
#         # var_nncanc0+=(list_nncanc[item][:][0]-average_nocanc0)**2/runs
#         # var_nncanc1+=(list_nncanc[item][:][1]-average_nocanc1)**2/runs
#         var_nocanc+=(GMI_nocanc[item]-average_nocanc)**2/runs
#         var_divcanc+=(GMI_divcanc[item]-average_divcanc)**2/runs 
#         var_nncanc+=(GMI_nncanc[item]-average_nncanc)**2/runs 


plt.figure("NN Cancellation", figsize=(3,3)) #figsize=(3.5,2)
for item in range(runs):
    plt.plot(GMI_nncanc[item],c=color_list[5],linewidth=1 ,alpha=0.9)
plt.plot(average_nncanc, c=color_list[4],linewidth=2, label="NN cancellation")
#plt.plot(average_nncanc1, c=color_list[10],linewidth=3, label="Enc"+str(1)+" NN cancellation")
#plt.fill_between(np.arange(num_epochs), average_nncanc+var_nncanc,average_nncanc-var_nncanc, color=color_list[4], alpha=0.2)
#plt.fill_between(np.arange(num_epochs), average_nncanc1+var_nncanc1,average_nncanc1-var_nncanc1, color=color_list[10], alpha=0.2)

#plt.title("Training GMIs for NN cancellation")
plt.legend(loc=3)
#plt.yscale('log')
plt.ylabel('GMI')
plt.grid()
plt.ylim(0,4)
plt.tight_layout()
plt.savefig('Multiple_NOMA/cancell_compare_GMI_NN_bw_modradius.pgf')

plt.show()



