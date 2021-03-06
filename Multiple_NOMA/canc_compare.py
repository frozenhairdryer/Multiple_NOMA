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
    'font.size' : 8,
})

### parameters
runs = 50
num_epochs = 60

sigma_n=torch.tensor([0.18,0.18])
M=torch.tensor([4,4])
#alph=torch.tensor([1,1/3*np.sqrt(2)])
alph=torch.tensor([1,1])

params=[runs,num_epochs,sigma_n,M,alph]

## load if already simulated
GMI_nocanc,GMI_divcanc,GMI_nncanc, best_impl, best_achieved, params = pickle.load( open( "cancel_compare.pkl", "rb" ) )
#GMI_nncanc, best_impl, best_achieved, params = pickle.load( open( "cancel_compare.pkl", "rb" ) )

num_epochs= params[1]
runs=params[0]
#GMI_nocanc=torch.zeros(num_epochs, runs)
#GMI_divcanc=torch.zeros(num_epochs, runs)
""" ## no canceller
GMI_nocanc=[]
list_nocanc=[]
#compare_data.append([])
for item in range(runs):
    _,en, dec, gmi, ser, gmi_exact, snr, const =Multipl_NOMA(M,sigma_n,train_params=[num_epochs,600,0.005],canc_method='none', modradius=alph, plotting=False)
    g = gmi_exact.detach().cpu().numpy()
    GMI_nocanc.append(np.sum(g, axis=1))
    list_nocanc.append(ser)
    if item==0:
        best_impl=[np.sum(g, axis=1),en, dec, ser]
        best_achieved='none'
    elif max(np.sum(g, axis=1))>max(best_impl[0]):
        best_impl=[np.sum(g, axis=1),en, dec, ser]
        best_achieved='none' 
    #exec(open("nn_enc_dec.py").read())
    #list_nocanc.append(validation_SERs)
    #compare_data[0].append(dir())
    #plt.close('all')


## division canceller 
GMI_divcanc=[]
list_divcanc=[]
#compare_data.append([])
for item in range(runs):
    try:
        #exec(open("nn_enc_dec_divcanc.py").read())
        _,en, dec, gmi, ser, gmi_exact, snr, const = Multipl_NOMA(M,sigma_n,train_params=[num_epochs,600,0.005],canc_method='div', modradius=alph, plotting=False)
        list_divcanc.append(ser)
        g = gmi_exact.detach().cpu().numpy()
        GMI_divcanc.append(np.sum(g, axis=1))
        if max(np.sum(g, axis=1))>max(best_impl[0]):
            best_impl=[np.sum(g, axis=1),en, dec, ser]
            best_achieved='div'
    except:
        print("Diverges!")
        GMI_divcanc.append(np.ones((num_epochs)))
    #compare_data[1].append(dir())
    #plt.close('all')

## NN canceller 
GMI_nncanc=[]
list_nncanc=[]
#compare_data.append([])
for item in range(runs):
    #compare_data[2].append(dir())
    #plt.close('all')
    _,en, dec,canc, gmi, ser, gmi_exact, snr, const = Multipl_NOMA(M,sigma_n,train_params=[num_epochs,600,0.003],canc_method='nn', modradius=alph, plotting=False)
    g = gmi_exact.detach().cpu().numpy()
    GMI_nncanc.append(np.sum(g, axis=1))
    list_nncanc.append(ser)
    if max(np.sum(g, axis=1))>max(best_impl[0]):
        best_impl=[np.sum(g, axis=1),en, dec, canc, ser]
        best_achieved='nn'

## save all data if further processing is wanted
with open('Multiple_NOMA/cancel_compare_gmiexact_freecanc_bw.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([GMI_nocanc,GMI_divcanc,GMI_nncanc, best_impl, best_achieved, params], f)
print("Best implementation achieved with "+best_achieved+" cancellation.")
 """
## figures
cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
#new_color_list = [[t/4 + 0.75 for t in color_list[k]] for k in range(len(color_list))]


#print(np.shape(np.array(GMI_nocanc)))






for item in range(runs):
    if item==0:
        # average_nocanc0=1/runs*list_nocanc[item][:][0]
        # average_nocanc1=1/runs*list_nocanc[item][:][1]
        # average_dcanc0=1/runs*list_divcanc[item][:][0]
        # average_dcanc1=1/runs*list_divcanc[item][:][1]
        # average_nncanc0=1/runs*list_nncanc[item][:][0]
        # average_nncanc1=1/runs*list_nncanc[item][:][1]
        average_nocanc = 1/runs*GMI_nocanc[item]
        average_divcanc = 1/runs*GMI_divcanc[item]
        average_nncanc = 1/runs*GMI_nncanc[item]
    else:
        # average_nocanc0+=1/runs*list_nocanc[item][:][0]
        # average_nocanc1+=1/runs*list_nocanc[item][:][1]
        # average_dcanc0+=1/runs*list_divcanc[item][:][0]
        # average_dcanc1+=1/runs*list_divcanc[item][:][1]
        # average_nncanc0+=1/runs*list_nncanc[item][:][0]
        # average_nncanc1+=1/runs*list_nncanc[item][:][1] 
        average_nocanc += 1/runs*GMI_nocanc[item]
        average_divcanc += 1/runs*GMI_divcanc[item]
        average_nncanc += 1/runs*GMI_nncanc[item]
    
#average_nocanc = np.mean(GMI_nocanc, axis=1)
#average_divcanc = np.mean(GMI_divcanc, axis=0)
#average_nncanc = np.mean(GMI_nncanc, axis=1)

#var_nocanc=np.var(GMI_nocanc, axis=1)
#var_divcanc=np.var(GMI_divcanc, axis=0)
#var_nncanc=np.var(GMI_nncanc)
#print(list_nocanc)
#print(np.min(list_nocanc, axis=1))

# variance:

""" for item in range(runs):
    if item==0:
        # var_nocanc0=(list_nocanc[item][:][0]-average_nocanc0)**2/runs
        # var_nocanc1=(list_nocanc[item][:][1]-average_nocanc1)**2/runs
        # var_dcanc0=(list_divcanc[item][:][0]-average_dcanc0)**2/runs
        # var_dcanc1=(list_divcanc[item][:][1]-average_dcanc1)**2/runs
        # var_nncanc0=(list_nncanc[item][:][0]-average_nocanc0)**2/runs
        # var_nncanc1=(list_nncanc[item][:][1]-average_nocanc1)**2/runs
        var_nocanc=(GMI_nocanc[item]-average_nocanc)**2/runs
        var_divcanc=(GMI_divcanc[item]-average_divcanc)**2/runs
        var_nncanc=(GMI_nncanc[item]-average_nncanc)**2/runs
    else:
        # var_nocanc0+=(list_nocanc[item][:][0]-average_nocanc0)**2/runs
        # var_nocanc1+=(list_nocanc[item][:][1]-average_nocanc1)**2/runs
        # var_dcanc0+=(list_divcanc[item][:][0]-average_dcanc0)**2/runs
        # var_dcanc1+=(list_divcanc[item][:][1]-average_dcanc1)**2/runs
        # var_nncanc0+=(list_nncanc[item][:][0]-average_nocanc0)**2/runs
        # var_nncanc1+=(list_nncanc[item][:][1]-average_nocanc1)**2/runs
        var_nocanc+=(GMI_nocanc[item]-average_nocanc)**2/runs
        var_divcanc+=(GMI_divcanc[item]-average_divcanc)**2/runs 
        var_nncanc+=(GMI_nncanc[item]-average_nncanc)**2/runs  """


""" plt.figure("No Cancellation", figsize=(3,3))
for item in range(runs):
    #plt.plot(list_nocanc[item][0],c=color_list[1], alpha=0.9)
    #plt.plot(list_nocanc[item][1],c=color_list[3], alpha=0.9)

    #plt.plot(list_divcanc[item][0], c=color_list[5], alpha=0.9)
    #plt.plot(list_divcanc[item][1], c=color_list[7], alpha=0.9)

    #plt.plot(list_nncanc[item][0],c=color_list[9], alpha=0.9)
    #plt.plot(list_nncanc[item][1],c=color_list[11], alpha=0.9)

    plt.plot(GMI_nocanc[item],c=color_list[1],linewidth=1, alpha=0.9)
plt.plot(average_nocanc, c=color_list[0],linewidth=2, label="no cancellation")
#plt.plot(average_nocanc1, c=color_list[2],linewidth=3, label="Enc"+str(1)+" no cancellation")
#plt.fill_between(np.arange(num_epochs), average_nocanc+var_nocanc,average_nocanc-var_nocanc, color=color_list[0], alpha=0.2)
#plt.fill_between(np.arange(num_epochs), average_nocanc1+var_nocanc1,average_nocanc1-var_nocanc1, color=color_list[2], alpha=0.2)

#plt.title("Training GMIs without cancellation")
plt.legend(loc=3)
#plt.yscale('log')
plt.ylabel('GMI')
plt.grid()
plt.ylim(0,4)
plt.tight_layout()
plt.savefig('Multiple_NOMA/cancell_compare_GMI_Nocanc_bw.pgf')


plt.figure("Division Cancellation", figsize=(3,3))
for item in range(runs):
    plt.plot(GMI_divcanc[item],c=color_list[3],linewidth=1 ,alpha=0.9)

plt.plot(average_divcanc, c=color_list[2], linewidth=2, label="division cancellation")
#plt.plot(average_dcanc1, c=color_list[6], linewidth=3, label="Enc"+str(1)+" division cancellation")

#plt.fill_between(np.arange(num_epochs), average_divcanc+var_divcanc,average_divcanc-var_divcanc, color=color_list[2], alpha=0.2)
#plt.fill_between(np.arange(num_epochs), average_dcanc1+var_dcanc1,average_dcanc1-var_dcanc1, color=color_list[6], alpha=0.2)

#plt.title("Training GMIs for Division cancellation")
plt.legend(loc=3)
#plt.yscale('log')
plt.ylabel('GMI')
plt.ylim(0,4)
plt.grid()
plt.tight_layout()
plt.savefig('Multiple_NOMA/cancell_compare_GMI_Divcanc_bw.pgf')


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
plt.savefig('Multiple_NOMA/cancell_compare_GMI_NNlogcanc_bw.pgf')

plt.show() """

plt.figure("Canc_compare", figsize=(6,2.5))
ax1 = plt.subplot(131)
for item in range(runs):
    plt.plot(GMI_nocanc[item],c=color_list[1],linewidth=1, alpha=0.9)
plt.plot(average_nocanc, c=color_list[0],linewidth=2, label="No cancellation")
plt.ylabel('GMI')
plt.xlabel('epochs')
plt.xlim(0,60)
plt.ylim(0,4)
plt.grid()
plt.legend(loc=3)

ax2 = plt.subplot(132, sharey=ax1, sharex=ax1)
for item in range(runs):
    plt.plot(GMI_nncanc[item],c=color_list[3],linewidth=1, alpha=0.9)
plt.plot(average_nncanc, c=color_list[2],linewidth=2, label="NN cancellation")
#plt.ylabel('GMI')
plt.xlabel('epochs')
plt.setp(ax2.get_yticklabels(), visible=False)
plt.grid()
plt.legend(loc=3)

ax3 = plt.subplot(133, sharey=ax1, sharex=ax1)
for item in range(runs):
    plt.plot(GMI_divcanc[item],c=color_list[5],linewidth=1, alpha=0.9)
plt.plot(average_divcanc, c=color_list[4],linewidth=2, label="Division cancellation")
#plt.ylabel('GMI')
plt.xlabel('epochs')
plt.grid()
plt.legend(loc=3)
plt.setp(ax3.get_yticklabels(), visible=False)

plt.tight_layout()
plt.savefig("cancel_comp_all_bw.pdf")

