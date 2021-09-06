import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from training_routine import *
import pickle

matplotlib.rcParams.update({
"pgf.texsystem": "pdflatex",
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
'font.size' : 10,
})

### parameters
runs = 50
num_epochs=40

sigma_n=torch.tensor([0.18,0.18])
M=torch.tensor([4,4])
#alph=torch.tensor([1,1/2*np.sqrt(2)])
alph1=torch.tensor([1,1])
#alph=[1,1]

params=[runs,num_epochs,sigma_n,M,alph1]


## NN canceller 
GMI_best=[]
modr=[]
#compare_data.append([])
for item in range(runs):
    #compare_data[2].append(dir())
    #plt.close('all')
    _,en, dec, mod, ser, gmi_exact = Multipl_NOMA(M,sigma_n,train_params=[num_epochs,600,0.005],canc_method='div', modradius=alph1, plotting=False)
    g = gmi_exact.detach().cpu().numpy()
    GMI_best.append(np.max(np.sum(g, axis=1)))
    modr.append(mod)
    if item==0:
        best_impl=[np.sum(g, axis=1),en, dec, ser]
        best_achieved='modr = '+str(mod)
    elif max(np.sum(g, axis=1))>max(best_impl[0]):
        best_impl=[np.sum(g, axis=1),en, dec, ser]
        best_achieved='modr = '+str(mod)

    

## save all data if further processing is wanted
with open('Multiple_NOMA/modr_eval.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([GMI_best,modr, best_impl, best_achieved, params], f)
print("Best implementation achieved with "+best_achieved+" cancellation.")

## figures
cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
#new_color_list = [[t/4 + 0.75 for t in color_list[k]] for k in range(len(color_list))]


#print(np.shape(np.array(GMI_nocanc)))
fig, axs = plt.subplots(1,2, figsize=(6, 3), sharey=True, sharex=True)

for x in range(runs):
    axs[0].scatter(modr[x][0], GMI_best[x], color=color_list[0])
    axs[1].scatter(modr[x][1], GMI_best[x], color=color_list[0])
axs[0].set_xlabel(r'effective modradius user 1')
axs[1].set_xlabel(r'effective modradius user 2')
axs[0].set_ylabel(r'GMI')
axs[1].set_ylabel(r'GMI')

axs[0].set_ylim(0,4)
axs[0].set_xlim(0,1.1)

axs[0].grid('-')
axs[1].grid('-')

plt.tight_layout()

plt.savefig("modr_eval.pgf")




