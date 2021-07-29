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
alph1=torch.tensor([1,1])
#alph=[1,1]

params=[runs,num_epochs,sigma_n,M,alph]


## NN canceller 
GMI_freecanc=[]
list_freecanc=[]
#compare_data.append([])
for item in range(runs):
    #compare_data[2].append(dir())
    #plt.close('all')
    _,en, dec,canc, gmi, ser, gmi_exact = Multipl_NOMA(M,sigma_n,train_params=[num_epochs,300,0.002],canc_method='nn', modradius=alph1, plotting=False)
    g = gmi_exact.detach().cpu().numpy()
    GMI_freecanc.append(np.sum(g, axis=1))
    list_freecanc.append(ser)
    if item==0:
        best_impl=[np.sum(g, axis=1),en, dec, canc, ser]
        best_achieved='nn'
    elif max(np.sum(g, axis=1))>max(best_impl[0]):
        best_impl=[np.sum(g, axis=1),en, dec, canc, ser]
        best_achieved='free'

GMI_dpcanc=[]
list_dpcanc=[]
#compare_data.append([])
for item in range(runs):
    #compare_data[2].append(dir())
    #plt.close('all')
    _,en, dec,canc, gmi, ser, gmi_exact = Multipl_NOMA(M,sigma_n,train_params=[num_epochs,300,0.002],canc_method='nn', modradius=alph, plotting=False)
    g = gmi_exact.detach().cpu().numpy()
    GMI_dpcanc.append(np.sum(g, axis=1))
    list_dpcanc.append(ser)
    if max(np.sum(g, axis=1))>max(best_impl[0]):
        best_impl=[np.sum(g, axis=1),en, dec, canc, ser]
        best_achieved='dp'

## save all data if further processing is wanted
with open('Multiple_NOMA/best_impl_freevsdp.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([best_impl, best_achieved, params], f)
with open('Multiple_NOMA/gmis_compare_freevsdp.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([GMI_freecanc,GMI_dpcanc, best_impl, best_achieved, params], f)
print("Best implementation achieved with "+best_achieved+" cancellation.")

## figures
cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
#new_color_list = [[t/4 + 0.75 for t in color_list[k]] for k in range(len(color_list))]


#print(np.shape(np.array(GMI_nocanc)))
plt.figure("Free Learning", figsize=(3.5,2))

for item in range(runs):
    #plt.plot(list_nocanc[item][0],c=color_list[1], alpha=0.9)
    #plt.plot(list_nocanc[item][1],c=color_list[3], alpha=0.9)

    #plt.plot(list_divcanc[item][0], c=color_list[5], alpha=0.9)
    #plt.plot(list_divcanc[item][1], c=color_list[7], alpha=0.9)

    #plt.plot(list_nncanc[item][0],c=color_list[9], alpha=0.9)
    #plt.plot(list_nncanc[item][1],c=color_list[11], alpha=0.9)

    plt.plot(GMI_freecanc[item],c=color_list[1],linewidth=1, alpha=0.9)



for item in range(runs):
    if item==0:
        average_freecanc = 1/runs*GMI_freecanc[item]
        average_dpcanc = 1/runs*GMI_dpcanc[item]

    else:
        average_freecanc += 1/runs*GMI_freecanc[item]
        average_dpcanc += 1/runs*GMI_dpcanc[item]
    

# variance:

for item in range(runs):
    if item==0:
        var_freecanc=(GMI_freecanc[item]-average_freecanc)**2/runs
        var_dpcanc=(GMI_dpcanc[item]-average_dpcanc)**2/runs
    else:
        var_freecanc+=(GMI_freecanc[item]-average_freecanc)**2/runs
        var_dpcanc+=(GMI_dpcanc[item]-average_dpcanc)**2/runs 



plt.plot(average_freecanc, c=color_list[0],linewidth=2)
#plt.plot(average_nocanc1, c=color_list[2],linewidth=3, label="Enc"+str(1)+" no cancellation")
plt.fill_between(np.arange(num_epochs), average_freecanc+var_freecanc,average_freecanc-var_freecanc, color=color_list[0], alpha=0.2)
#plt.fill_between(np.arange(num_epochs), average_nocanc1+var_nocanc1,average_nocanc1-var_nocanc1, color=color_list[2], alpha=0.2)

#plt.title("Training GMIs without cancellation")
#plt.legend(loc=4)
#plt.yscale('log')
plt.ylabel('GMI')
plt.grid()
plt.ylim(0,4)
plt.tight_layout()
plt.savefig('Multiple_NOMA/free_learning.pgf')


plt.figure("Design Proposal", figsize=(3.5,2))
for item in range(runs):
    plt.plot(GMI_dpcanc[item],c=color_list[3],linewidth=1 ,alpha=0.9)

plt.plot(average_dpcanc, c=color_list[2], linewidth=2)
#plt.plot(average_dcanc1, c=color_list[6], linewidth=3, label="Enc"+str(1)+" division cancellation")

plt.fill_between(np.arange(num_epochs), average_dpcanc+var_dpcanc,average_dpcanc-var_dpcanc, color=color_list[2], alpha=0.2)
#plt.fill_between(np.arange(num_epochs), average_dcanc1+var_dcanc1,average_dcanc1-var_dcanc1, color=color_list[6], alpha=0.2)

#plt.title("Training GMIs for Division cancellation")
#plt.legend(loc=4)
#plt.yscale('log')
plt.ylabel('GMI')
plt.ylim(0,4)
plt.grid()
plt.tight_layout()
plt.savefig('Multiple_NOMA/design_prop.pgf')




plt.show()



