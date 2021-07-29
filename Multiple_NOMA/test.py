from training_routine import *
import pickle

M=torch.tensor([4,4], dtype=int)
sigma_n=torch.tensor([0.08,0.08], dtype=float)
runs = 50
num_epochs=60
#begin_time = datetime.datetime.now()
## NN canceller 
GMI_nncanc=[]
list_nncanc=[]
#compare_data.append([])
for item in range(runs):
    #compare_data[2].append(dir())
    #plt.close('all')
    _,en, dec,canc, gmi, ser, gmi_exact = Multipl_NOMA(M,sigma_n,train_params=cp.array([60,300,0.002]),canc_method='nn', modradius=torch.tensor([1,1/3*np.sqrt(2)]), plotting=False)
    g = gmi_exact.detach().cpu().numpy()
    GMI_nncanc.append(np.sum(g, axis=1))
    list_nncanc.append(ser)
    if item==0:
        best_impl=[np.sum(g, axis=1),en, dec, canc, ser]
        best_achieved='nn'
    elif max(np.sum(g, axis=1))>max(best_impl[0]):
        best_impl=[np.sum(g, axis=1),en, dec, canc, ser]
        best_achieved='nn'

## figures
cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors

for item in range(runs):
    if item==0:
        average_nncanc = 1/runs*GMI_nncanc[item]
    else:
        average_nncanc += 1/runs*GMI_nncanc[item]
    

# variance:

for item in range(runs):
    if item==0:
        var_nncanc=(GMI_nncanc[item]-average_nncanc)**2/runs
    else:
        var_nncanc+=(GMI_nncanc[item]-average_nncanc)**2/runs 

#print(datetime.datetime.now() - begin_time)                                     
plt.figure("NN Cancellation", figsize=(3.5,2))
for item in range(runs):
    plt.plot(GMI_nncanc[item],c=color_list[5],linewidth=0.5 ,alpha=0.9)
plt.plot(average_nncanc, c=color_list[4],linewidth=2, label="NN cancellation")
#plt.plot(average_nncanc1, c=color_list[10],linewidth=3, label="Enc"+str(1)+" NN cancellation")
plt.fill_between(np.arange(num_epochs), average_nncanc+var_nncanc,average_nncanc-var_nncanc, color=color_list[4], alpha=0.2)
#plt.fill_between(np.arange(num_epochs), average_nncanc1+var_nncanc1,average_nncanc1-var_nncanc1, color=color_list[10], alpha=0.2)

#plt.title("Training GMIs for NN cancellation")
plt.legend(loc=3)
#plt.yscale('log')
plt.ylabel('GMI')
plt.grid()
plt.ylim(0,4)
plt.tight_layout()
plt.savefig('cancell_compare_GMI_NNlogcanc_modradius.pgf')

#sigma_n=torch.tensor([0.03,0.03,0.03])
#M=torch.tensor([4,4,4])
#alph=[np.sqrt(2)/9,1/3*np.sqrt(2),1]

#_,enc, dec,canc, gmi, ser, gmi_exact = Multipl_NOMA(M,sigma_n,train_params=[200,1200,0.002],canc_method='nn', modradius=alph, plotting=False)

#with open('3user_modraduis.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([ enc, dec, canc, gmi, ser, gmi_exact], f)
