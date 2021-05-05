# Code adapted from: 
# L. Schmalen, M. L. Schmid, and B. Geiger, "Machine Learning and Optimization in Communications - Lecture Examples," available online at http://www.github.org/KIT-CEL/lecture-examples/, 2019


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
rng = np.random.default_rng()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

###############################################################
### Parameters ###
#lsym  = 1                      # length of symbol in samples
#h_in  = 1                      # pulseshaping filter
#Csize = [4]                    # size of constellation diagrams [4,4,8] means first sender has 4 bit, second sender 4 and third sender 8 bit
#s_off = [0]                    # sample offset -> model imperfect synchronization

#train_size = 20                # number of symbols in the training set
#test_size = 10                 # number of symbols in the test data
#batch_size = 5

# Training parameters
num_epochs = 30
batches_per_epoch = 300

# number of symbols
M = np.array([2,8,2])
M_all = np.product(M)

# Definition of noise
EbN0 = np.array([60,60,18])

# validation set. Training examples are generated on the fly
N_valid = 10000

###############################################################

# helper function to compute the symbol error rate
def SER(predictions, labels):
    s2=np.argmax(predictions, 1)
    return (np.sum(s2 != labels) / predictions.shape[0])



# noise standard deviation
sigma_n = np.sqrt((1/2/np.log2(M)) * 10**(-EbN0/10))


# number of neurons in hidden layers at transmitter
hidden_neurons_TX_1 = 50
hidden_neurons_TX_2 = 50
hidden_neurons_TX_3 = 50
hidden_neurons_TX_4 = 50
hidden_neurons_TX = [hidden_neurons_TX_1, hidden_neurons_TX_2, hidden_neurons_TX_3, hidden_neurons_TX_4]

# number of neurons in hidden layers at receiver
hidden_neurons_RX_1 = 50
hidden_neurons_RX_2 = 50
hidden_neurons_RX_3 = 50
hidden_neurons_RX_4 = 50
hidden_neurons_RX = [hidden_neurons_RX_1, hidden_neurons_RX_2, hidden_neurons_RX_3, hidden_neurons_RX_4]

# Generate Validation Data
y_valid = np.random.randint(M_all,size=N_valid)
y_valid_onehot = np.eye(M_all)[y_valid]


# meshgrid for plotting
ext_max = 2  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
mgx,mgy = np.meshgrid(np.linspace(-ext_max,ext_max,400), np.linspace(-ext_max,ext_max,400))
meshgrid = np.column_stack((np.reshape(mgx,(-1,1)),np.reshape(mgy,(-1,1)))) 

class Encoder(nn.Module):
    def __init__(self,M, hidden_neurons_TX):
        super(Encoder, self).__init__()
        # Define Transmitter Layer: Linear function, M input neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(M,hidden_neurons_TX[0]) 
        self.fcT2 = nn.Linear(hidden_neurons_TX[0], hidden_neurons_TX[1]) 
        self.fcT3 = nn.Linear(hidden_neurons_TX[1], hidden_neurons_TX[2]) 
        self.fcT4 = nn.Linear(hidden_neurons_TX[2], hidden_neurons_TX[3]) 
        self.fcT5 = nn.Linear(hidden_neurons_TX[3], 2) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        encoded = self.network_transmitter(x)
        # compute normalization factor and normalize channel output             
        norm_factor = torch.sqrt(torch.mean(torch.mul(encoded,encoded)) * 2 ) 
        modulated = encoded / norm_factor
        return modulated
        
    def network_transmitter(self,batch_labels):
        out = self.activation_function(self.fcT1(batch_labels))
        out = self.activation_function(self.fcT2(out))
        out = self.activation_function(self.fcT3(out))
        out = self.activation_function(self.fcT4(out))
        out = self.activation_function(self.fcT5(out))
        return out
    

class Decoder(nn.Module):
    def __init__(self,M_all, hidden_neurons_RX):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 input neurons (real and imaginary part), M output neurons (symbols)
        self.fcR1 = nn.Linear(2,hidden_neurons_RX[0]) 
        self.fcR2 = nn.Linear(hidden_neurons_RX[0], hidden_neurons_RX[1]) 
        self.fcR3 = nn.Linear(hidden_neurons_RX[1], hidden_neurons_RX[2]) 
        self.fcR4 = nn.Linear(hidden_neurons_RX[2], hidden_neurons_RX[3]) 
        self.fcR5 = nn.Linear(hidden_neurons_RX[3], M_all) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        logits = self.network_receiver(x)
        return logits
    
    def network_receiver(self,inp):
        out = self.activation_function(self.fcR1(inp))
        out = self.activation_function(self.fcR2(out))
        out = self.activation_function(self.fcR3(out))
        out = self.activation_function(self.fcR4(out))
        logits = self.activation_function(self.fcR5(out))
        return logits
    

enc=[]
optimizer=[]
for const in range(np.size(M)):
    enc.append(Encoder(M[const],hidden_neurons_TX))
    enc[const].to(device)
    # Adam Optimizer
    optimizer.append(optim.Adam(enc[const].parameters()))




dec_1 = Decoder(M_all,hidden_neurons_RX)
dec_1.to(device)
#dec_2.to(device)

softmax = nn.Softmax(dim=1)

# Cross Entropy loss
loss_fn = nn.CrossEntropyLoss()

# Adam Optimizer
#optimizer = [optim.Adam(enc_1.parameters()), optim.Adam(enc_2.parameters()), optim.Adam(dec_1.parameters()) ]
optimizer.append(optim.Adam(dec_1.parameters()))

#enc_1 = enc[0]
#enc_2 = enc[1]

# Vary batch size during training
batch_size_per_epoch = np.linspace(100,10000,num=num_epochs)

validation_SERs = np.zeros(num_epochs)
validation_received = []
decision_region_evolution = []
constellations = []
# constellations only used for plotting
constellation_base = []
for modulator in range(np.size(M)):
    constellation_base.append([])

print('Start Training')
for epoch in range(num_epochs):
    
    batch_labels = torch.empty(int(batch_size_per_epoch[epoch]),np.size(M), device=device)
    #batch_labels_2 = torch.empty(int(batch_size_per_epoch[epoch]), device=device)

    for step in range(batches_per_epoch):
        # Generate training data: In most cases, you have a dataset and do not generate a training dataset during training loop
        # sample new mini-batch directory on the GPU (if available)
        M_enc=1
        for num in range(np.size(M)):        
            batch_labels[:,num].random_(M[num])
            batch_labels_onehot = torch.zeros(int(batch_size_per_epoch[epoch]), M[num], device=device)
            batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels[:,num].long()]=1
            
            if num==0:
                # Propagate (training) data through the first transmitter
                modulated = enc[0](batch_labels_onehot)
                # Propagate through channel 1
                received = torch.add(modulated, sigma_n[num]*torch.randn(len(modulated),2).to(device))
            else:
                
                modulated = torch.view_as_real(torch.view_as_complex(received)*torch.view_as_complex(enc[num](batch_labels_onehot)))
                batch_labels[:,num] = batch_labels[:,num-1]+M_enc*batch_labels[:,num]
                received = torch.add(modulated, sigma_n[num]*torch.randn(len(modulated),2).to(device))
            
            if num==np.size(M)-1:
                decoded = dec_1(received)
            M_enc = M_enc*M[num]


        # Resulting information labels
        new_labels=batch_labels[:,np.size(M)-1]
        new_labels_onehot = torch.zeros(int(batch_size_per_epoch[epoch]), M_all, device=device)
        new_labels_onehot[range(new_labels_onehot.shape[0]), new_labels.long()]=1

        # compute loss
        loss = loss_fn(decoded.squeeze(), new_labels.long())

        # compute gradients
        loss.backward()
        
        # Adapt weights
        for elem in optimizer:
            elem.step()
        
        # reset gradients
        for elem in optimizer:
            elem.zero_grad()




    # compute validation SER
    M_enc=1
    for num in range(np.size(M)):
        if num==0:
            y = np.mod(y_valid,M[0])
            y_onehot = np.eye(M[0])[y]
            encoded = enc[0](torch.Tensor(y_onehot).to(device))
            #y_val = (y_valid/M[0]).astype(int)
            y_val = y_valid-y
            ycontrol = y
            channel = torch.add(encoded, sigma_n[0]*torch.randn(len(encoded),2).to(device))
        elif num<np.size(M)-1:
            y = np.array(np.mod(y_val, M_enc*M[num])/M_enc, int)
            y_onehot = np.eye(M[num])[y]
            encoded = torch.view_as_real(torch.view_as_complex(channel)*torch.view_as_complex(enc[num](torch.Tensor(y_onehot).to(device))))
            channel = torch.add(encoded, sigma_n[num]*torch.randn(len(encoded),2).to(device))
            y_val = (y_val-y*M_enc)
            ycontrol= ycontrol + y*M_enc
        else:
            y = np.array(y_val/M_enc, int)
            y_onehot = np.eye(M[num])[y]
            encoded = torch.view_as_real(torch.view_as_complex(channel)*torch.view_as_complex(enc[num](torch.Tensor(y_onehot).to(device))))
            channel = torch.add(encoded, sigma_n[num]*torch.randn(len(encoded),2).to(device))
            ycontrol = ycontrol + y*M_enc
        M_enc=M_enc*M[num]

    if ycontrol.all()!=y_valid.all():
        print("CONVERSION ERROR")
    #y1 = np.mod(y_valid,M[0])
    #y1_onehot = np.eye(M[0])[y1]

    #y2 = (y_valid/M[0]).astype(int)
    #y2_onehot = np.eye(M[1])[y2]

    #encoded_1 = enc[0](torch.Tensor(y1_onehot).to(device))
    #ch1 = torch.add(encoded_1, sigma_n[0]*torch.randn(len(encoded_1),2).to(device))
    #encoded_2 = torch.view_as_real(torch.view_as_complex(ch1)*torch.view_as_complex(enc[1](torch.Tensor(y2_onehot).to(device))))
    #ch2 = torch.add(encoded_2, sigma_n[1]*torch.randn(len(encoded_2),2).to(device))
    decoded = dec_1(channel)

    out_valid = softmax(decoded)
    validation_SERs[epoch] = SER(out_valid.detach().cpu().numpy().squeeze(), y_valid)
    print('Validation SER after epoch %d: %f (loss %1.8f)' % (epoch, validation_SERs[epoch], loss.detach().cpu().numpy()))                
    
    validation_received.append(channel.detach().cpu().numpy())
    
    # calculate and store base constellations
    for num in range(np.size(M)):
        constellation_base[num].append(torch.view_as_complex(enc[num].network_transmitter(torch.eye(M[num]))))
        #constellation_base[1].append(torch.view_as_complex(enc[0].network_transmitter(torch.eye(M[0]))))
        if num==0:
            encoded = constellation_base[num][epoch].repeat(M[num+1])
            encoded = torch.reshape(encoded,(M[num+1],int(encoded.size()/M[num+1])))
            #print(encoded)
        elif num<np.size(M)-1:
            helper = torch.reshape(constellation_base[num][epoch].repeat(M[num]),(M[num], int(constellation_base[num][epoch].size()[0])))
            encoded = torch.matmul(torch.transpose(encoded,0,1),helper).flatten().repeat(M[num+1])/M[num]
            encoded = torch.reshape(encoded,(M[num+1],int(encoded.size()/M[num+1])))
        else:
            helper = torch.reshape(constellation_base[num][epoch].repeat(M[num]),(M[num], int(constellation_base[num][epoch].size()[0])))
            encoded = torch.matmul(torch.transpose(encoded,0,1),helper).flatten()/M[num]


    #encoded=torch.zeros(list(M))+0j
    #for encoder_item in range(M[0]):
    #    for encoder_item2 in range(M[1]):
    #        encoded[encoder_item, encoder_item2]=constellation_base[0][epoch][encoder_item]*constellation_base[1][epoch][encoder_item2]

    #encoded=np.array(const).flatten()
    constellations.append(encoded.detach().cpu().numpy())
        
    # store decision region for generating the animation
    mesh_prediction = softmax(dec_1.network_receiver(torch.Tensor(meshgrid).to(device)))
    decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
    
print('Training finished')

cmap = matplotlib.cm.tab20
base = plt.cm.get_cmap(cmap)
color_list = base.colors
new_color_list = [[t/2 + 0.5 for t in color_list[k]] for k in range(len(color_list))]

# find minimum SER from validation set
min_SER_iter = np.argmin(validation_SERs)
ext_max_plot = 1.05*np.max(np.abs(validation_received[min_SER_iter]))

plt.figure(figsize=(6,6))
font = {'size'   : 14}
plt.rc('font', **font)
plt.rc('text', usetex=True)
    
plt.plot(validation_SERs,marker='.',linestyle='--')
plt.scatter(min_SER_iter,validation_SERs[min_SER_iter],marker='o',c='red')
plt.annotate('Min', (0.95*min_SER_iter,1.4*validation_SERs[min_SER_iter]),c='red')
plt.xlabel('epoch no.',fontsize=14)
plt.ylabel('SER',fontsize=14)
plt.grid(which='both')
plt.title('SER on Validation Dataset',fontsize=16)

print('Minimum SER obtained: %1.5f (epoch %d out of %d)' % (validation_SERs[min_SER_iter], min_SER_iter, len(validation_SERs)))
print('The corresponding constellation symbols are:\n', constellations[min_SER_iter])


plt.figure(figsize=(19,6))
font = {'size'   : 14}
plt.rc('font', **font)
plt.rc('text', usetex=True)


plt.subplot(131)
plt.scatter(np.real(constellations[min_SER_iter].flatten()),np.imag(constellations[min_SER_iter].flatten()),c=range(M_all), cmap='tab20',s=50)
plt.axis('scaled')
plt.xlabel(r'$\Re\{r\}$',fontsize=14)
plt.ylabel(r'$\Im\{r\}$',fontsize=14)
plt.xlim((-ext_max_plot,ext_max_plot))
plt.ylim((-ext_max_plot,ext_max_plot))
plt.grid(which='both')
plt.title('Constellation',fontsize=16)

val_cmplx=validation_received[min_SER_iter][:,0]+1j*validation_received[min_SER_iter][:,1]


plt.subplot(132)
#plt.contourf(mgx,mgy,decision_region_evolution[-1].reshape(mgy.shape).T,cmap='coolwarm',vmin=0.3,vmax=0.7)
#plt.scatter(validation_received[min_SER_iter][:,0], validation_received[min_SER_iter][:,1], c=y_valid, cmap='tab20',s=4)
plt.scatter(np.real(val_cmplx), np.imag(val_cmplx), c=y_valid, cmap='tab20',s=4)
plt.axis('scaled')
plt.xlabel(r'$\Re\{r\}$',fontsize=14)
plt.ylabel(r'$\Im\{r\}$',fontsize=14)
plt.xlim((-ext_max_plot,ext_max_plot))
plt.ylim((-ext_max_plot,ext_max_plot))
plt.title('Received',fontsize=16)

plt.subplot(133)
decision_scatter = np.argmax(decision_region_evolution[min_SER_iter], 1)
plt.scatter(meshgrid[:,0], meshgrid[:,1], c=decision_scatter, cmap=matplotlib.colors.ListedColormap(colors=new_color_list),s=4)
#plt.scatter(validation_received[min_SER_iter][0:4000,0], validation_received[min_SER_iter][0:4000,1], c=y_valid[0:4000], cmap='tab20',s=4)
plt.scatter(np.real(val_cmplx[0:4000]), np.imag(val_cmplx[0:4000]), c=y_valid[0:4000], cmap='tab20',s=4)
plt.axis('scaled')
plt.xlim((-ext_max_plot,ext_max_plot))
plt.ylim((-ext_max_plot,ext_max_plot))
plt.xlabel(r'$\Re\{r\}$',fontsize=14)
plt.ylabel(r'$\Im\{r\}$',fontsize=14)
plt.title('Decision regions',fontsize=16)

plt.figure("base constellations")
for num in range(np.size(M)):
    plt.subplot(1,np.size(M),num+1)
    plt.scatter(np.real(constellation_base[num][min_SER_iter].detach().numpy()),np.imag(constellation_base[num][min_SER_iter].detach().numpy()), c=np.arange(M[num]))
    plt.xlim((-ext_max_plot,ext_max_plot))
    plt.ylim((-ext_max_plot,ext_max_plot))
    plt.xlabel(r'$\Re\{r\}$',fontsize=14)
    plt.ylabel(r'$\Im\{r\}$',fontsize=14)
    plt.grid()
    plt.tight_layout()

#plt.subplot(122)
#plt.scatter(np.real(constellation_base[1][min_SER_iter].detach().numpy()),np.imag(constellation_base[1][min_SER_iter].detach().numpy()),c=np.arange(M[1]))
#plt.xlim((-ext_max_plot,ext_max_plot))
#plt.ylim((-ext_max_plot,ext_max_plot))
#plt.xlabel(r'$\Re\{r\}$',fontsize=14)
#plt.ylabel(r'$\Im\{r\}$',fontsize=14)
#plt.tight_layout()



plt.show()
#plt.savefig('decision_region_AWGN_AE_EbN0%1.1f_M%d.pdf' % (EbN0,M), bbox_inches='tight')