import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.show(block=False)
import pickle
### parameters
runs = 10
compare_data=[]

## no canceller
list_nocanc=[]
compare_data.append([])
for item in range(runs):
    exec(open("nn_enc_dec.py").read())
    list_nocanc.append(validation_SERs)
    compare_data[0].append(dir())
    plt.close('all')

## division canceller 
list_divcanc=[]
compare_data.append([])
for item in range(runs):
    try:
        exec(open("nn_enc_dec_divcanc.py").read())
        list_divcanc.append(validation_SERs)
    except:
        list_divcanc.append(np.ones((runs,2)))
    compare_data[1].append(dir())
    plt.close('all')

## NN canceller 
list_nncanc=[]
compare_data.append([])
for item in range(runs):
    exec(open("nn_enc_dec_canc.py").read())
    list_nncanc.append(validation_SERs)
    compare_data[2].append(dir())
    plt.close('all')

## save all data if further processing is wanted
with open('cancel_compare_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(compare_data, f)
with open('cancel_compare_lists.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([list_nocanc,list_divcanc,list_nncanc], f)


## figures
cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
#new_color_list = [[t/4 + 0.75 for t in color_list[k]] for k in range(len(color_list))]


print(np.shape(np.array(list_nocanc)))
plt.figure(figsize=(7,3))

for item in range(runs):
    plt.plot(list_nocanc[item][0],c=color_list[1], alpha=0.9)
    plt.plot(list_nocanc[item][1],c=color_list[3], alpha=0.9)

    plt.plot(list_divcanc[item][0], c=color_list[5], alpha=0.9)
    plt.plot(list_divcanc[item][1], c=color_list[7], alpha=0.9)

    plt.plot(list_nncanc[item][0],c=color_list[9], alpha=0.9)
    plt.plot(list_nncanc[item][1],c=color_list[11], alpha=0.9)


for item in range(runs):
    if item==0:
        average_nocanc0=1/runs*list_nocanc[item][:][0]
        average_nocanc1=1/runs*list_nocanc[item][:][1]
        average_dcanc0=1/runs*list_divcanc[item][:][0]
        average_dcanc1=1/runs*list_divcanc[item][:][1]
        average_nncanc0=1/runs*list_nncanc[item][:][0]
        average_nncanc1=1/runs*list_nncanc[item][:][1]
    else:
        average_nocanc0+=1/runs*list_nocanc[item][:][0]
        average_nocanc1+=1/runs*list_nocanc[item][:][1]
        average_dcanc0+=1/runs*list_divcanc[item][:][0]
        average_dcanc1+=1/runs*list_divcanc[item][:][1]
        average_nncanc0+=1/runs*list_nncanc[item][:][0]
        average_nncanc1+=1/runs*list_nncanc[item][:][1]

print(list_nocanc)
print(np.min(list_nocanc, axis=1))

plt.plot(average_nocanc0, c=color_list[0],linewidth=3, label="Enc"+str(0)+" no cancellation")
plt.plot(average_nocanc1, c=color_list[2],linewidth=3, label="Enc"+str(1)+" no cancellation")


plt.plot(average_dcanc0, c=color_list[4], linewidth=3, label="Enc"+str(0)+" division cancellation")
plt.plot(average_dcanc1, c=color_list[6], linewidth=3, label="Enc"+str(1)+" division cancellation")


plt.plot(average_nncanc0, c=color_list[8],linewidth=3, label="Enc"+str(0)+" NN cancellation")
plt.plot(average_nncanc1, c=color_list[10],linewidth=3, label="Enc"+str(1)+" NN cancellation")

plt.title("Training SERs for different cancellation approaches")
plt.legend(loc=1)
plt.yscale('log')
plt.ylabel('log(SER)')
plt.grid()
plt.tight_layout()
plt.savefig('cancell_compare.png')
plt.show()

