from os import error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Encoder(nn.Module):
    def __init__(self,M, alph):
        super(Encoder, self).__init__()
        # Define Transmitter Layer: Linear function, M input neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(M,2*M) 
        self.fcT2 = nn.Linear(2*M, 2*M) 
        self.fcT5 = nn.Linear(2*M, 2)
        self.alpha=alph 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        out = self.activation_function(self.fcT1(x))
        out = self.activation_function(self.fcT2(out))
        encoded = self.activation_function(self.fcT5(out))
        # compute normalization factor and normalize channel output
        #norm_factor = torch.mean(torch.abs(torch.view_as_complex(encoded)).flatten()) # normalize mean amplitude to 1
        norm_factor = torch.max(torch.abs(torch.view_as_complex(encoded)).flatten())        
        #norm_factor = torch.sqrt(torch.mean(torch.mul(encoded,encoded)) * 2 ) # normalize mean amplitude in real and imag to sqrt(1/2)
        modulated = encoded / norm_factor
        if self.alpha!=1:
            modulated = torch.view_as_real((1+self.alpha*torch.view_as_complex(modulated))/(1+self.alpha))
        return modulated
    

class Decoder(nn.Module):
    def __init__(self,M):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 input neurons (real and imaginary part), M output neurons (symbols)
        self.fcR1 = nn.Linear(2,2*M) 
        self.fcR2 = nn.Linear(2*M,2*M) 
        self.fcR5 = nn.Linear(2*M, M) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        out = self.activation_function(self.fcR1(x))
        out = self.activation_function(self.fcR2(out))
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
        out = self.activation_function(self.fcR1(inp))
        out += self.activation_function(self.fcR2(decout))
        #out = self.activation_function(self.fcR3(out))
        out = self.activation_function(self.fcR4(out))
        logits = self.activation_function(self.fcR5(out))
        return logits
    
def SER(predictions, labels):
    s2=np.argmax(predictions, 1)
    return (np.sum(s2 != labels) / predictions.shape[0])

def MI(predictions, labels):
    px=torch.bincount(torch.tensor(labels))/np.size(labels)
    py=torch.sum(predictions, axis=0)/np.size(labels)
    pxy = torch.zeros((len(px),len(px)))
    for elem in range(len(labels)):
        pxy[labels[elem]] += predictions[elem]/len(labels)
    
    info=pxy*torch.log2(pxy/(py*px))
    for elemx in range(len(px)):
        for elemy in range(len(px)):
            info[elemx,elemy]=pxy[elemx,elemy]*torch.log2(pxy[elemx,elemy]/(py[elemx]*px[elemy]))
    infos=sum(sum(info))
    return infos

def Multipl_NOMA(M=4,sigma_n=0.1,train_params=[50,300,0.005],canc='none', modradius=1, plotting=True):
    # train_params=[num_epochs,batches_per_epoch, learn_rate]
    # canc is the chosen cancellation method:
    # division cancellation: canc='div'
    # no cancellation: canc='none'
    # cancellation with neural network: canc='nn'
    # modradius is the permitted signal amplitude for each encoder
    # M, sigma_n, modradius are lists of the same size
    if len(M)!=len(sigma_n) or len(M)!=len(modradius):
        raise error("M, sigma_n, modradius need to be of same size!")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs=train_params[0]
    batches_per_epoch=train_params[1]
    learn_rate =train_params[2]
    N_valid=10000
    weight=np.ones(len(M))

    # Generate Validation Data
    rng = np.random.default_rng()
    y_valid = rng.integers(0,M,size=(N_valid,np.size(M)))

    if plotting==True:
        # meshgrid for plotting
        ext_max = 2  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
        mgx,mgy = np.meshgrid(np.linspace(-ext_max,ext_max,400), np.linspace(-ext_max,ext_max,400))
        meshgrid = np.column_stack((np.reshape(mgx,(-1,1)),np.reshape(mgy,(-1,1))))
    
    enc=[]
    dec=[]
    optimizer=[]

    if canc=='none' or canc=='div':
        for const in range(np.size(M)):
            enc.append(Encoder(M[const], modradius[const]))
            dec.append(Decoder(M[const]))
            enc[const].to(device)
            # Adam Optimizer
            #optimizer.append([])
            optimizer.append(optim.Adam(enc[const].parameters(), lr=learn_rate))
            optimizer.append(optim.Adam(dec[const].parameters(), lr=learn_rate))
    
    elif canc=='nn':
        canc = []
        for const in range(np.size(M)):
            enc.append(Encoder(M[const]))
            dec.append(Decoder(M[const]))
            enc[const].to(device)
            # Adam Optimizer
            #optimizer.append([])
            if const>0:
                canc.append(Canceller(np.product(M)))
                optimizer.append(optim.Adam(canc[const-1].parameters(),lr=learn_rate))
            optimizer.append(optim.Adam(enc[const].parameters(),lr=learn_rate))
            optimizer.append(optim.Adam(dec[const].parameters(),lr=learn_rate))

    else:
        raise error("Cancellation method invalid. Choose 'none','nn', or'div'")

    softmax = nn.Softmax(dim=1)

    # Cross Entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # Vary batch size during training
    batch_size_per_epoch = np.linspace(100,10000,num=num_epochs)

    validation_SERs = np.zeros((np.size(M),num_epochs))
    validation_GMI = np.zeros((np.size(M),num_epochs))
    validation_received = []
    if plotting==True:
        decision_region_evolution = []
        constellations = []
        # constellations only used for plotting
        constellation_base = []

        for modulator in range(np.size(M)):
            constellation_base.append([])

    print('Start Training')
    GMI = np.zeros(num_epochs)
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
                
                if num==0:
                    # Propagate (training) data through the first transmitter
                    modulated = enc[0](batch_labels_onehot)
                    # Propagate through channel 1
                    received = torch.add(modulated, sigma_n[num]*torch.randn(len(modulated),2).to(device))
                else:
                    modulated = torch.view_as_real(torch.view_as_complex(received)*((torch.view_as_complex(enc[num](batch_labels_onehot)))))
                    received = torch.add(modulated, sigma_n[num]*torch.randn(len(modulated),2).to(device))
                
                if num==np.size(M)-1:
                    if canc=='none':
                        for dnum in range(np.size(M)):
                            decoded.append(dec[dnum](received))
                    
                    elif canc=='div':
                        for dnum in range(np.size(M)):
                            if dnum==0:
                                decoded[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](received))
                                cancelled = torch.view_as_real(torch.view_as_complex(received)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                            else:
                                decoded[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled))
                                cancelled =  torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                    
                    elif canc=='nn':
                        for dnum in range(np.size(M)):
                            if dnum==0:
                                decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](received)
                                cancelled=(canc[dnum](received,enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                            elif dnum==np.size(M)-1:
                                decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled)
                            else:
                                decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled)
                                
                                cancelled=(canc[dnum](received,enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
            # Calculate Loss
            for num in range(np.size(M)):
                # calculate loss as weighted addition of losses for each enc[x] to dec[x] path
                if num==0:
                    loss = weight[0]*loss_fn(decoded[0], batch_labels[:,0].long())
                else:
                    loss += weight[num]*loss_fn(decoded[num], batch_labels[:,num].long())

            # compute gradients
            loss.backward()
            
            # Adapt weights
            for elem in optimizer:
                elem.step()
            
            # reset gradients
            for elem in optimizer:
                elem.zero_grad()

        # compute validation SER, SNR, GMI
        if plotting==True:
            cvalid = np.zeros(N_valid)
        decoded_valid=[]
        SNR = np.zeros(np.size(M))
        for num in range(np.size(M)):
            y_valid_onehot = np.eye(M[num])[y_valid[:,num]]
            
            if num==0:
                encoded = enc[num](torch.Tensor(y_valid_onehot).to(device))
                SNR[num] = 20*torch.log10(torch.mean(torch.abs(encoded))/sigma_n[num])
                channel = torch.add(encoded, sigma_n[num]*torch.randn(len(encoded),2).to(device))
                # color map for plot
                if plotting==True:
                    cvalid=y_valid[:,num]
            else:
                encoded = torch.view_as_real(torch.view_as_complex(channel)*(torch.view_as_complex(enc[num](torch.Tensor(y_valid_onehot).to(device)))))
                #sigma = sigma_n[num]*np.mean(np.abs(torch.view_as_complex(encoded).detach().numpy()))
                SNR[num] = 20*torch.log10(torch.mean(torch.abs(encoded))/sigma_n[num])
                channel = torch.add(encoded, sigma_n[num]*torch.randn(len(encoded),2).to(device))
                #color map for plot
                if plotting==True:
                    cvalid= cvalid+M[num]*y_valid[:,num]
            if num==np.size(M)-1 and canc=='none':
                    for dnum in range(np.size(M)):
                        decoded_valid.append(dec[dnum](channel))
            if num==np.size(M)-1 and canc=='div':
                for dnum in range(np.size(M)):
                    if dnum==0:
                        decoded_valid[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](channel)
                        cancelled = torch.view_as_real(torch.view_as_complex(channel)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))
                    else:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled))
                        cancelled = torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))
            if num==np.size(M)-1 and canc=='nn':
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

        for num in range(len(M)):
            GMI[epoch] += MI(softmax(decoded_valid[num]), y_valid[:,num])
            validation_SERs[num][epoch] = SER(softmax(decoded_valid[num]).detach().cpu().numpy().squeeze(), y_valid[:,num])
            print('Validation SER after epoch %d for encoder %d: %f (loss %1.8f)' % (epoch,num, validation_SERs[num][epoch], loss.detach().cpu().numpy()))                
            if validation_SERs[num][epoch]>1/M[num] and epoch>5:
                #Weight is increased, when error probability is higher than symbol probability -> misclassification 
                weight[num] += 1
        weight=weight/np.sum(weight)*np.size(M) # normalize weight sum 
        #print("weight set to "+str(weight))
        print("GMI is: "+ str(GMI[epoch]) + " bit")
        print("SNR is: "+ str(SNR)+" dB")
        if epoch==0:
            enc_best=enc
            dec_best=dec
            if canc!='none':
                canc_best=canc
            best_epoch=0
        elif GMI[epoch]>GMI[best_epoch]:
            enc_best=enc
            dec_best=dec
            if canc!='none':
                canc_best=canc
            best_epoch=0

        if np.sum(validation_SERs[:,epoch])<0.2:
            #set weights back if under threshold
            weight=np.ones(np.size(M))
        
        validation_received.append(channel.detach().cpu().numpy())

        if plotting==True:
            for num in range(np.size(M)):
                constellation_base[num].append(torch.view_as_complex(enc[num](torch.eye(M[num]))))
                #constellation_base[1].append(torch.view_as_complex(enc[0].network_transmitter(torch.eye(M[0]))))
                if num==0:
                    encoded = constellation_base[num][epoch].repeat(M[num+1])/M[num+1]
                    encoded = torch.reshape(encoded,(M[num+1],int(np.size(encoded.detach().cpu().numpy())/M[num+1])))
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
                decision_region_evolution.append([])
                mesh_prediction = softmax(dec[num](torch.Tensor(meshgrid).to(device)))
                decision_region_evolution[num].append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
            
    print('Training finished')
    if plotting==True:
        plot_training(validation_SERs, validation_received,cvalid,M, constellations, GMI) #TODO define plot_training
    max_GMI = np.argmax(GMI)
    if canc=='div' or canc=='nn':
        return(enc_best,dec_best,canc_best)
    else:
        return(enc_best,dec_best)

def plot_training(SERs,valid_r,cvalid,M, const, GMIs):
    cmap = matplotlib.cm.tab20
    base = plt.cm.get_cmap(cmap)
    color_list = base.colors
    new_color_list = [[t/2 + 0.5 for t in color_list[k]] for k in range(len(color_list))]

    sum_SERs = np.sum(SERs, axis=0)/np.size(M)
    min_SER_iter = np.argmin(np.sum(SERs,axis=0))
    max_GMI = np.argmax(GMIs)
    ext_max_plot = 1.05*np.max(np.abs(valid_r[min_SER_iter]))

    plt.figure("SERs",figsize=(6,6))
    font = {'size'   : 14}
    plt.rc('font', **font)
    #plt.rc('text', usetex=True)
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    })

    for num in range(np.size(M)):
        plt.plot(SERs[num],marker='.',linestyle='--', label="Enc"+str(num))
        plt.scatter(min_SER_iter,SERs[num][min_SER_iter],marker='o',c='red')
        plt.annotate('Min', (0.95*min_SER_iter,1.4*SERs[num][min_SER_iter]),c='red')
    plt.xlabel('epoch no.',fontsize=14)
    plt.ylabel('SER',fontsize=14)
    plt.grid(which='both')
    plt.legend(loc=1)
    plt.title('SER on Validation Dataset',fontsize=16)
    plt.tight_layout()

    plt.figure("GMIs",figsize=(6,6))
    plt.plot(GMIs,marker='.',linestyle='--')
    plt.scatter(max_GMI,GMIs[max_GMI],marker='o',c='red')
    plt.annotate('Max', (0.95*max_GMI,1.4*GMIs[max_GMI]),c='red')
    plt.xlabel('epoch no.',fontsize=14)
    plt.ylabel('GMI',fontsize=14)
    plt.grid(which='both')
    plt.title('GMI on Validation Dataset',fontsize=16)
    plt.tight_layout()

    plt.figure("constellation")
    plt.subplot(121)
    plt.scatter(np.real(const[min_SER_iter].flatten()),np.imag(const[min_SER_iter].flatten()),c=range(np.product(M)), cmap='tab20',s=50)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$',fontsize=14)
    plt.ylabel(r'$\Im\{r\}$',fontsize=14)
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid(which='both')
    plt.title('Constellation',fontsize=16)

    val_cmplx=valid_r[min_SER_iter][:,0]+1j*valid_r[min_SER_iter][:,1]


    plt.subplot(122)
    plt.scatter(np.real(val_cmplx), np.imag(val_cmplx), c=cvalid, cmap='tab20',s=4)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$',fontsize=14)
    plt.ylabel(r'$\Im\{r\}$',fontsize=14)
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.title('Received',fontsize=16)
    print('Minimum mean SER obtained: %1.5f (epoch %d out of %d)' % (sum_SERs[min_SER_iter], min_SER_iter, len(SERs[0])))
    print('Maximum obtained GMI: %1.5f (epoch %d out of %d)' % (GMIs[max_GMI],max_GMI,len(GMIs)))
    print('The corresponding constellation symbols are:\n', const[min_SER_iter])
    
    plt.show()


#Multipl_NOMA(M=[4,4],sigma_n=[0.1,0.1],train_params=[10,300,0.005],canc='none', modradius=[1,1], plotting=True)