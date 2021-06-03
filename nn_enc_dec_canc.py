# Code adapted from: 
# L. Schmalen, M. L. Schmid, and B. Geiger, "Machine Learning and Optimization in Communications - Lecture Examples," available online at http://www.github.org/KIT-CEL/lecture-examples/, 2019

for element in dir():
    if element[0:2] != "__":
        if element[0:4] != "list" and  element!="runs" and element!="compare_data": 
            del globals()[element]

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
rng = np.random.default_rng()

torch.autograd.set_detect_anomaly(True) # find problems computing the gradient

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

###############################################################
### Parameters ###
#lsym  = 1                      # length of symbol in samples
#h_in  = 1                      # pulseshaping filter
#s_off = [0]                    # sample offset -> model imperfect synchronization

# Training parameters
num_epochs = 50
#random_epochs = 30
batches_per_epoch = 300
learn_rate =0.005

# number of symbols
M = np.array([4,4])
M_all = np.product(M)

# Definition of noise
EbN0 = np.array([16,14])

# loss weights for training: Change if one Encoder is more important:
weight=np.ones(np.size(M))
# weights are changed during training if one encoder gets stuck

# validation set. Training examples are generated on the fly
N_valid = 10000

###############################################################

# helper function to compute the symbol error rate
def SER(predictions, labels):
    s2=np.argmax(predictions, 1)
    return (np.sum(s2 != labels) / predictions.shape[0])



# noise standard deviation
sigma_n = np.sqrt((1/2/np.log2(M)) * 10**(-EbN0/10))

SNR = EbN0+10*np.log10(np.log2(M))
print("Channel SNR is: "+str(SNR)+" dB")

# Generate Validation Data
y_valid = rng.integers(0,M,size=(N_valid,np.size(M)))


# meshgrid for plotting
ext_max = 1.5  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
mgx,mgy = np.meshgrid(np.linspace(-ext_max,ext_max,400), np.linspace(-ext_max,ext_max,400))
meshgrid = np.column_stack((np.reshape(mgx,(-1,1)),np.reshape(mgy,(-1,1)))) 

class Encoder(nn.Module):
    def __init__(self,Mod):
        super(Encoder, self).__init__()
        # Define Transmitter Layer: Linear function, M input neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(Mod,2*Mod) 
        self.fcT2 = nn.Linear(2*Mod, 2*Mod)
        #self.fcT3 = nn.Linear(2*Mod, 2*Mod) 
        #self.fcT4 = nn.Linear(2*Mod, 2*Mod) 
        self.fcT5 = nn.Linear(2*Mod, 2) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        encoded = self.network_transmitter(x)
        # compute normalization factor and normalize channel output
        #norm_factor = torch.mean(torch.abs(torch.view_as_complex(encoded)).flatten()) # normalize mean amplitude to 1
        norm_factor = torch.max(torch.abs(torch.view_as_complex(encoded)).flatten()) # normalize max amplitude to 1 -> somehow results in psk?         
        #norm_factor = torch.sqrt(torch.mean(torch.mul(encoded,encoded)) * 2 ) # normalize mean amplitude in real and imag to sqrt(1/2)
        modulated = encoded / norm_factor
        return modulated
        

    def network_transmitter(self,batch_labels):
        out = self.activation_function(self.fcT1(batch_labels))
        out = self.activation_function(self.fcT2(out))
        #out = self.activation_function(self.fcT3(out))
        #out = self.activation_function(self.fcT4(out))
        out = self.activation_function(self.fcT5(out))
        return out
    

class Decoder(nn.Module):
    def __init__(self,Mod):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 input neurons (real and imaginary part), M output neurons (symbols)
        self.fcR1 = nn.Linear(2,2*Mod) 
        self.fcR2 = nn.Linear(2*Mod,2*Mod) 
        self.fcR3 = nn.Linear(2*Mod,2*Mod) 
        #self.fcR4 = nn.Linear(2*Mod,2*Mod) 
        self.fcR5 = nn.Linear(2*Mod, Mod) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        logits = self.network_receiver(x)
        return logits
    
    def network_receiver(self,inp):
        out = self.activation_function(self.fcR1(inp))
        #out = self.activation_function(self.fcR2(out))
        out = self.activation_function(self.fcR3(out))
        #out = self.activation_function(self.fcR4(out))
        logits = self.activation_function(self.fcR5(out))
        return logits


class Canceller(nn.Module):
    def __init__(self,Mod):
        super(Canceller, self).__init__()
        self.fcR1 = nn.Linear(2,Mod) 
        self.fcR2 = nn.Linear(2,Mod) 
        self.fcR3 = nn.Linear(Mod,Mod) 
        self.fcR4 = nn.Linear(Mod,Mod) 
        self.fcR5 = nn.Linear(Mod, 2) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x, decoutput):
        # compute output
        logits = self.cancellation(x, decoutput)
        #norm_factor = torch.max(torch.abs(torch.view_as_complex(logits)).flatten())
        norm_factor = torch.mean(torch.abs(torch.view_as_complex(logits)).flatten()) # normalize mean amplitude to 1
        logits = logits/norm_factor
        return logits
    
    def cancellation(self,inp, decout):
        #print(self.activation_function(self.fcR1(inp)))
        #print(self.activation_function(self.fcR2(decout)))
        out = self.activation_function(self.fcR1(inp))
        out += self.activation_function(self.fcR2(decout))
        #out = self.activation_function(self.fcR2(out))
        #out = self.activation_function(self.fcR3(out))
        out = self.activation_function(self.fcR4(out))
        logits = self.activation_function(self.fcR5(out))
        return logits

# Collect encoders, decoders and optimizer in list
enc=[]
dec=[]
canc = []
optimizer=[]
for const in range(np.size(M)):
    
    enc.append(Encoder(M[const]))
    dec.append(Decoder(M[const]))
    enc[const].to(device)
    # Adam Optimizer
    #optimizer.append([])
    if const>0:
            canc.append(Canceller(M_all))
            optimizer.append(optim.Adam(canc[const-1].parameters(),lr=learn_rate))
    optimizer.append(optim.Adam(enc[const].parameters(),lr=learn_rate))
    optimizer.append(optim.Adam(dec[const].parameters(),lr=learn_rate))


softmax = nn.Softmax(dim=1)

# Cross Entropy loss
loss_fn = nn.CrossEntropyLoss()



# Vary batch size during training
batch_size_per_epoch = np.linspace(100,10000,num=num_epochs)

validation_SERs = np.zeros((np.size(M),num_epochs))
validation_received = []
decision_region_evolution = []
for num in range(np.size(M)):
    decision_region_evolution.append([])

constellations = []
cancelled=[]

# constellations only used for plotting
constellation_base = []
for modulator in range(np.size(M)):
    constellation_base.append([])

print('Start Training')
for epoch in range(num_epochs):
    batch_labels = torch.empty(int(batch_size_per_epoch[epoch]),np.size(M), device=device)
    
    for step in range(batches_per_epoch):
        # Generate training data: In most cases, you have a dataset and do not generate a training dataset during training loop
        # sample new mini-batch directory on the GPU (if available)
        decoded=[]
        for num in range(np.size(M)):        
            batch_labels[:,num].random_(M[num])
            batch_labels_onehot = torch.zeros(int(batch_size_per_epoch[epoch]), M[num], device=device)
            batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels[:,num].long()]=1
            decoded.append([])
            if num==0:
                # Propagate (training) data through the first transmitter
                modulated = enc[0](batch_labels_onehot)
                sigma = sigma_n[num]*np.mean(np.abs(torch.view_as_complex(modulated).detach().numpy()))
                # Propagate through channel 1
                received = torch.add(modulated, sigma*torch.randn(len(modulated),2).to(device))
            else:
                modulated = torch.view_as_real(torch.view_as_complex(received)*(torch.view_as_complex(enc[num](batch_labels_onehot))))
                sigma = sigma_n[num]*np.mean(np.abs(torch.view_as_complex(modulated).detach().numpy()))
                received = torch.add(modulated, sigma*torch.randn(len(modulated),2).to(device))
            
            if num==np.size(M)-1:
                for dnum in range(np.size(M)):
                    if dnum==0:
                        decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](received)
                        cancelled_train=(canc[dnum](received,enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                    elif dnum==np.size(M)-1:
                        decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled_train)
                        #decoded.append(dec[np.size(M)-dnum-1](cancelled_train))
                    else:
                        decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled_train)
                        #decoded.append(dec[np.size(M)-dnum-1](cancelled_train))
                        cancelled_train=(canc[dnum](received,enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                    

        # Calculate Loss
        for num in range(np.size(M)):
            if num==0:
                loss = weight[0]*loss_fn(decoded[0], batch_labels[:,0].long())
            else:
                loss += weight[num]*loss_fn(decoded[num], batch_labels[:,num].long())
                #loss -= loss_fn(decoded[num], batch_labels[:,num-1].long()) # maximize cross-Entropy of canceller output (here further: decoder output) and already decoded signal
            
        # compute gradients
        
        loss.backward()
        
        # Adapt weights
        for elem in optimizer:
            elem.step()
        
        # reset gradients
        for elem in optimizer:
            elem.zero_grad()




    # compute validation SER
    cvalid = np.zeros(N_valid)
    decoded_valid=[]
    cancelled.append([])
    for num in range(np.size(M)):
        y_valid_onehot = np.eye(M[num])[y_valid[:,num]]
        decoded_valid.append([])
        if num==0:
            encoded = enc[num](torch.Tensor(y_valid_onehot).to(device))
            sigma = sigma_n[num]*np.mean(np.abs(torch.view_as_complex(encoded).detach().numpy()))
            channel = torch.add(encoded, sigma*torch.randn(len(encoded),2).to(device))
            # color map for plot
            cvalid=y_valid[:,num]
        else:
            encoded = torch.view_as_real(torch.view_as_complex(channel)*(torch.view_as_complex(enc[num](torch.Tensor(y_valid_onehot).to(device)))))
            sigma = sigma_n[num]*np.mean(np.abs(torch.view_as_complex(encoded).detach().numpy()))
            channel = torch.add(encoded, sigma*torch.randn(len(encoded),2).to(device))
            #color map for plot
            cvalid= cvalid+M[num]*y_valid[:,num]
        if num==np.size(M)-1:
                for dnum in range(np.size(M)):
                    if dnum==0:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](channel))
                        # canceller
                        cancelled[epoch].append(canc[dnum](channel,enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))
                    elif dnum==np.size(M)-1:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled[epoch][dnum-1]))
                    else:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled[epoch][dnum-1]))
                        cancelled[epoch].append(canc[dnum](received,enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))

    

    for num in range(np.size(M)):
        out_valid = softmax(decoded_valid[num])
        #print(batch_labels[:,num])
        validation_SERs[num][epoch] = SER(out_valid.detach().cpu().numpy().squeeze(), y_valid[:,num])
        print('Validation SER after epoch %d for encoder %d: %f (loss %1.8f)' % (epoch,num, validation_SERs[num][epoch], loss.detach().cpu().numpy()))
        #if validation_SERs[num][epoch]>1/M[num] and epoch>5:
            #Weight is increased, when error probability is higher than symbol probability -> misclassification 
            #weight[num] += 1
            
    #weight=weight/np.sum(weight)*np.size(M) # normalize weight sum
    #print("weight changed to "+str(weight))
    
    #if np.sum(validation_SERs[:,epoch])<0.2:
    #    weight=np.ones(np.size(M))

    
    validation_received.append(channel.detach().cpu().numpy())
    
    # calculate and store base constellations
    for num in range(np.size(M)):
        constellation_base[num].append(torch.view_as_complex(enc[num](torch.eye(M[num]))))
        #constellation_base[1].append(torch.view_as_complex(enc[0].network_transmitter(torch.eye(M[0]))))
        if num==0:
            encoded = constellation_base[num][epoch].repeat(M[num+1])/M[num+1]
            encoded = torch.reshape(encoded,(M[num+1],int(encoded.size()/M[num+1])))
            #print(encoded)
        elif num<np.size(M)-1:
            helper = torch.reshape(constellation_base[num][epoch].repeat(M[num]),(M[num], int(constellation_base[num][epoch].size()[0])))
            encoded = torch.matmul(torch.transpose(encoded,0,1),helper).flatten().repeat(M[num+1])/M[num+1]
            encoded = torch.reshape(encoded,(M[num+1],int(encoded.size()/M[num+1])))
        else:
            helper = torch.reshape(constellation_base[num][epoch].repeat(M[num]),(M[num], int(constellation_base[num][epoch].size()[0])))
            encoded = torch.matmul(torch.transpose(encoded,0,1),helper).flatten()


    constellations.append(encoded.detach().cpu().numpy())
        
    # store decision region for generating the animation
    for num in range(np.size(M)):
        #decision_region_evolution.append([])
        if num==0:
            mesh_prediction = softmax(dec[np.size(M)-num-1](torch.Tensor(meshgrid).to(device)))
        else:
            mesh_prediction = softmax(dec[np.size(M)-num-1](canc[num-1](torch.Tensor(meshgrid).to(device),enc[np.size(M)-num](mesh_prediction))))
        decision_region_evolution[num].append(0.195*mesh_prediction.detach().cpu().numpy() +0.4)
    
print('Training finished')

cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
new_color_list = [[t/2 + 0.5 for t in color_list[k]] for k in range(len(color_list))]

# find minimum SER from validation set
#min_SER_iter = np.argmin(validation_SERs)
# minimize sum of SER:
sum_SERs = np.sum(validation_SERs, axis=0)/np.size(M)
min_SER_iter = np.argmin(np.sum(validation_SERs,axis=0))
ext_max_plot = 1.05*np.max(np.abs(validation_received[min_SER_iter]))

plt.figure(figsize=(6,6))
font = {'size'   : 14}
plt.rc('font', **font)
plt.rc('text', usetex=True)
    
    
for num in range(np.size(M)):
    plt.plot(validation_SERs[num],marker='.',linestyle='--', label="Enc"+str(num))
    plt.scatter(min_SER_iter,validation_SERs[num][min_SER_iter],marker='o',c='red')
    plt.annotate('Min', (0.95*min_SER_iter,1.4*validation_SERs[num][min_SER_iter]),c='red')
plt.xlabel('epoch no.',fontsize=14)
plt.ylabel('SER',fontsize=14)
plt.grid(which='both')
plt.legend(loc=1)
plt.title('SER on Validation Dataset',fontsize=16)

print('Minimum SER obtained: %1.5f (epoch %d out of %d)' % (sum_SERs[min_SER_iter], min_SER_iter, len(validation_SERs[0])))
print('The corresponding constellation symbols are:\n', constellations[min_SER_iter])


plt.figure(figsize=(14,6))
font = {'size'   : 14}
plt.rc('font', **font)
plt.rc('text', usetex=True)


plt.subplot(121)
plt.scatter(np.real(constellations[min_SER_iter].flatten()),np.imag(constellations[min_SER_iter].flatten()),c=range(M_all), cmap='tab20',s=50)
plt.axis('scaled')
plt.xlabel(r'$\Re\{r\}$',fontsize=14)
plt.ylabel(r'$\Im\{r\}$',fontsize=14)
plt.xlim((-ext_max_plot,ext_max_plot))
plt.ylim((-ext_max_plot,ext_max_plot))
plt.grid(which='both')
plt.title('Constellation',fontsize=16)

val_cmplx=validation_received[min_SER_iter][:,0]+1j*validation_received[min_SER_iter][:,1]


plt.subplot(122)
plt.scatter(np.real(val_cmplx), np.imag(val_cmplx), c=cvalid, cmap='tab20',s=4)
plt.axis('scaled')
plt.xlabel(r'$\Re\{r\}$',fontsize=14)
plt.ylabel(r'$\Im\{r\}$',fontsize=14)
plt.xlim((-ext_max_plot,ext_max_plot))
plt.ylim((-ext_max_plot,ext_max_plot))
plt.title('Received',fontsize=16)

plt.figure("Decision regions", figsize=(14,6))
for num in range(np.size(M)):
    plt.subplot(1,np.size(M),num+1)
    decision_scatter = np.argmax(decision_region_evolution[num][min_SER_iter], 1)
    if num==0:
        plt.scatter(meshgrid[:,0], meshgrid[:,1], c=decision_scatter,s=4,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:M[num]]))
        plt.scatter(np.real(val_cmplx[0:4000]), np.imag(val_cmplx[0:4000]), c=cvalid[0:4000], cmap='tab20',s=4)
    else:
        #vals=cancelled[min_SER_iter][num-1][:,0].detach().numpy()+1j*cancelled[min_SER_iter][num-1][:,1].detach().numpy()
        plt.scatter(meshgrid[:,0], meshgrid[:,1], c=decision_scatter,s=4,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[M[num-1]:M[num-1]+M[num]]))
        #plt.scatter(np.real(vals[0:4000]), np.imag(vals[0:4000]), c=cvalid[0:4000], cmap='tab20',s=4)
        plt.scatter(np.real(val_cmplx[0:4000]), np.imag(val_cmplx[0:4000]), c=cvalid[0:4000], cmap='tab20',s=4)

    #plt.scatter(validation_received[min_SER_iter][0:4000,0], validation_received[min_SER_iter][0:4000,1], c=y_valid[0:4000], cmap='tab20',s=4)
        
    plt.axis('scaled')
    #plt.xlim((-ext_max_plot,ext_max_plot))
    #plt.ylim((-ext_max_plot,ext_max_plot))
    plt.xlabel(r'$\Re\{r\}$',fontsize=14)
    plt.ylabel(r'$\Im\{r\}$',fontsize=14)
    plt.title('Decision regions for Decoder %d' % num,fontsize=16)
plt.tight_layout()

plt.figure("Base Constellations", figsize=(10,5))
for num in range(np.size(M)):
    plt.subplot(1,np.size(M),num+1)
    plt.scatter(np.real(constellation_base[num][min_SER_iter].detach().numpy()),np.imag(constellation_base[num][min_SER_iter].detach().numpy()), c=np.arange(M[num]))
    plt.xlim((-ext_max_plot,ext_max_plot))
    plt.ylim((-ext_max_plot,ext_max_plot))
    plt.xlabel(r'$\Re\{r\}$',fontsize=14)
    plt.ylabel(r'$\Im\{r\}$',fontsize=14)
    plt.grid()
    plt.tight_layout()


#plt.show()
#plt.savefig('decision_region_AWGN_AE_EbN0%1.1f_M%d.pdf' % (EbN0,M), bbox_inches='tight')