import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from functions import *


rng = np.random.default_rng()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
softmax = nn.Softmax(dim=1)

cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
new_color_list = [[t/2 + 0.5 for t in color_list[k]] for k in range(len(color_list))]



saved_data, best_method,data = pickle.load( open( "back_data/best_impl5050.pkl", "rb" ) )
#print(saved_data)
if best_method=='nn':
    gmi,enc, dec, canc, ser =saved_data
else:
    gmi,enc, dec, ser = saved_data

constellation_base=[]
decision_region_evolution=[]
M=[4,4]

ext_max = 2  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
mgx,mgy = np.meshgrid(np.linspace(-ext_max,ext_max,400), np.linspace(-ext_max,ext_max,400))
meshgrid = np.column_stack((np.reshape(mgx,(-1,1)),np.reshape(mgy,(-1,1))))

for num in range(len(enc)):
    constellation_base.append(torch.view_as_complex(enc[num](torch.eye(M[num]))))
    #constellation_base[1].append(torch.view_as_complex(enc[0].network_transmitter(torch.eye(M[0]))))
    if num==0:
        encoded = constellation_base[num].repeat(M[num+1])/M[num+1]
        encoded = torch.reshape(encoded,(M[num+1],int(np.size(encoded.detach().cpu().numpy())/M[num+1])))
        #print(encoded)
    elif num<np.size(M)-1:
        helper = torch.reshape(constellation_base[num].repeat(M[num]),(M[num], int(constellation_base[num].size()[0])))
        encoded = torch.matmul(torch.transpose(encoded,0,1),helper).flatten().repeat(M[num+1])/M[num+1]
        encoded = torch.reshape(encoded,(M[num+1],int(encoded.size()/M[num+1])))
    else:
        helper = torch.reshape(constellation_base[num].repeat(M[num]),(M[num], int(constellation_base[num].size()[0])))
        encoded = torch.matmul(torch.transpose(encoded,0,1),helper).flatten()

constellations=encoded.detach().cpu().numpy()

# store decision region for generating the animation
if best_method=='none':
    for num in range(np.size(M)):
        #decision_region_evolution.append([])
        mesh_prediction = softmax(dec[num](torch.Tensor(meshgrid).to(device)))
        decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
    
if best_method=='div':
    for num in range(np.size(M)):
        #decision_region_evolution.append([])
        if num==0:
            mesh_prediction = softmax(dec[np.size(M)-num-1](torch.Tensor(meshgrid).to(device)))
            canc_grid = torch.view_as_real(torch.view_as_complex(torch.Tensor(meshgrid).to(device))/torch.view_as_complex(enc[np.size(M)-num-1](mesh_prediction)))
        else:
            mesh_prediction = softmax(dec[np.size(M)-num-1](canc_grid))
            canc_grid = torch.view_as_real(torch.view_as_complex(canc_grid)/torch.view_as_complex(enc[np.size(M)-num-1](mesh_prediction)))
        decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)

plt.figure("Decision regions", figsize=(16,6))
for num in range(np.size(M)):
    plt.subplot(1,np.size(M),num+1)
    decision_scatter = np.argmax(decision_region_evolution[num], 1)
    if num==0:
        plt.scatter(meshgrid[:,0], meshgrid[:,1], c=decision_scatter,s=4,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:M[num]]))
    else:
        plt.scatter(meshgrid[:,0], meshgrid[:,1], c=decision_scatter,s=4,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[M[num-1]:M[num-1]+M[num]]))
    plt.scatter(np.real(constellations), np.imag(constellations),s=20,marker='x', cmap='tab20')
    #plt.scatter(np.real(val_cmplx[0:4000]), np.imag(val_cmplx[0:4000]), c=cvalid[0:4000], cmap='tab20',s=4)
    plt.axis('scaled')
    #plt.xlim((-ext_max_plot,ext_max_plot))
    #plt.ylim((-ext_max_plot,ext_max_plot))
    plt.xlabel(r'$\Re\{r\}$',fontsize=14)
    plt.ylabel(r'$\Im\{r\}$',fontsize=14)
    plt.title('Decision regions for Decoder %d' % num,fontsize=16)

plt.tight_layout()

plt.show()