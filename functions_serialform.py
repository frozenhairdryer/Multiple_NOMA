from os import error
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Encoder(nn.Module):
    def __init__(self,M, mradius):
        super(Encoder, self).__init__()
        # Define Transmitter Layer: Linear function, M input neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(M,2*M) 
        self.fcT2 = nn.Linear(2*M, 2*M)
        self.fcT3 = nn.Linear(2*M, 2*M) 
        self.fcT5 = nn.Linear(2*M, 2)
        self.modradius= mradius

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        out = self.activation_function(self.fcT1(x))
        out = self.activation_function(self.fcT2(out))
        out = self.activation_function(self.fcT3(out))
        encoded = self.activation_function(self.fcT5(out))
        # compute normalization factor and normalize channel output
        #norm_factor = torch.mean(torch.abs(torch.view_as_complex(encoded)).flatten()) # normalize mean amplitude to 1
        norm_factor = torch.max(torch.abs(torch.view_as_complex(encoded)).flatten())
        #if norm_factor>1:        
        #norm_factor = torch.sqrt(torch.mean(torch.mul(encoded,encoded)) * 2 ) # normalize mean amplitude in real and imag to sqrt(1/2)
        modulated = encoded / norm_factor
        #else:
        #    modulated = encoded
        if self.modradius!=1:
            modulated = torch.view_as_real((1+self.modradius*torch.view_as_complex(modulated))/(1+self.modradius))
        return modulated
    

class Decoder(nn.Module):
    def __init__(self,M):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 input neurons (real and imaginary part), M output neurons (symbols)
        self.fcR1 = nn.Linear(2,2*M) 
        self.fcR2 = nn.Linear(2*M,2*M) 
        self.fcR3 = nn.Linear(2*M,2*M) 
        self.fcR5 = nn.Linear(2*M, M) 
        #self.alpha=torch.tensor([alph,alph])
        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x):
        # compute output
        #n1 = torch.view_as_real((torch.view_as_complex(x)*(1+torch.view_as_complex(self.alpha))-1)/torch.view_as_complex(self.alpha))
        #n1 = (x*(self.alpha+1)-torch.tensor([1,0])).float()
        out = self.activation_function(self.fcR1(x))
        out = self.activation_function(self.fcR2(out))
        #out = self.activation_function(self.fcR3(out))
        
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
        out = self.activation_function(self.fcR3(out))
        out = self.activation_function(self.fcR4(out))
        logits = self.activation_function(self.fcR5(out))
        return logits
    
def SER(predictions, labels):
    s2=np.argmax(predictions, 1)
    return (np.sum(s2 != labels) / predictions.shape[0])

def BER(predictions, labels,m):
    # Bit representation of symbols
    binaries = torch.from_numpy(np.reshape(np.unpackbits(np.uint8(np.arange(0,m))), (-1,8))).float()
    binaries = binaries[:,int(8-np.log2(m)):]
    y_valid_binary = binaries[labels,:].detach().cpu().numpy()
    pred_binary = binaries[np.argmax(predictions,1),:].detach().cpu().numpy()
    ber=torch.zeros(int(np.log2(m)))
    for bit in range(int(np.log2(m))):
        ber[bit] = np.mean(1-np.isclose((pred_binary[:,bit] > 0.5).astype(float), y_valid_binary[:,bit]))
        if ber[bit]>0.5:
            ber[bit]=1-ber[bit]
    return ber, y_valid_binary,pred_binary


def GMI(SERs, M, ber=None):
    # gmi estimate or calculation, if bers are given
    M_all=np.product(M)
    gmi_est=0
    for mod in range(np.size(M)):
        #Pe = SERs[mod]/np.log2(M[mod])
        #Pe = 1-(1-SERs[mod])**(1/np.log2(M[mod]))
        #gmi_est+= np.log2(M[mod])*(Pe*np.log2(Pe/0.5+1e-12)+(1-Pe)*np.log2((1-Pe)/0.5+1e-12))
        Pe = SERs[mod] # only one bit contributes to errors
        gmi_est+= np.log2(M[mod])*(Pe*np.log2(Pe/0.5+1e-12)+(1-Pe)*np.log2((1-Pe)/0.5+1e-12))
    if ber!=None:
        gmi=[]
        for num in range(len(M)):
            for x in range(int(np.log2(M[num]))):
                gmi.append((1+(ber[num][x]*np.log2(ber[num][x]+1e-12)+(1-ber[num][x])*np.log2((1-ber[num][x])+1e-12))).detach().numpy())    
        return gmi_est, np.array(gmi)
    else:
        return gmi_est

def Multipl_NOMA(M=4,sigma_n=0.1,train_params=[50,300,0.005],canc_method='none', modradius=1, plotting=True, encoder=None):
    # train_params=[num_epochs,batches_per_epoch, learn_rate]
    # canc_method is the chosen cancellation method:
    # division cancellation: canc_method='div'
    # no cancellation: canc_method='none'
    # cancellation with neural network: canc_method='nn'
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
        ext_max = 1.5  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
        mgx,mgy = np.meshgrid(np.linspace(-ext_max,ext_max,70), np.linspace(-ext_max,ext_max,70))
        meshgrid = np.column_stack((np.reshape(mgx,(-1,1)),np.reshape(mgy,(-1,1))))
    
    if encoder==None:
        enc=[]
        dec=[]
        optimizer=[]

        if canc_method=='none' or canc_method=='div':
            for const in range(np.size(M)):
                enc.append(Encoder(M[const], modradius[const]))
                dec.append(Decoder(M[const]))
                enc[const].to(device)
                # Adam Optimizer
                #if const==0:
                #    optimizer=optim.Adam(enc[const].parameters(), lr=learn_rate)
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                #else:
                #    optimizer.add_param_group({'params':enc[const].parameters()})
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                optimizer.append([])
                optimizer[const].append(optim.Adam(enc[const].parameters(), lr=learn_rate))
                optimizer[const].append(optim.Adam(dec[const].parameters(), lr=learn_rate))
                
        
        elif canc_method=='nn':
            canc = []
            for const in range(np.size(M)):
                enc.append(Encoder(M[const],modradius[const]))
                dec.append(Decoder(M[const]))
                enc[const].to(device)
                # Adam Optimizer
                optimizer.append([])
                #if const==0:
                #    optimizer=optim.Adam(enc[const].parameters(), lr=learn_rate)
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                optimizer[const].append(optim.Adam(enc[const].parameters(),lr=learn_rate))
                optimizer[const].append(optim.Adam(dec[const].parameters(),lr=learn_rate))
                if const>0:
                    canc.append(Canceller(np.product(M)))
                #    optimizer.add_param_group({'params':enc[const].parameters()})
                #    optimizer.add_param_group({'params':canc[const-1].parameters()})
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                    optimizer[const].append(optim.Adam(canc[const-1].parameters(),lr=learn_rate))
    else:
        enc=encoder
        dec=[]
        optimizer=[]
        lhelp=len(encoder)

        if canc_method=='none' or canc_method=='div':
            for const in range(np.size(M)):
                optimizer.append([])
                if const>=lhelp:
                    enc.append(Encoder(M[const], modradius[const]))
                dec.append(Decoder(M[const]))
                enc[const].to(device)
                # Adam Optimizer
                #if const==0:
                #    optimizer=optim.Adam(enc[const].parameters(), lr=learn_rate)
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                #else:
                #    optimizer.add_param_group({'params':enc[const].parameters()})
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                if const<lhelp:
                    print("skip encoder optim for enc" + str(const))
                    #pass
                    optimizer[const].append(optim.Adam(enc[const].parameters(), lr=learn_rate*0.01))
                else:
                    optimizer[const].append(optim.Adam(enc[const].parameters(), lr=learn_rate))
                optimizer[const].append(optim.Adam(dec[const].parameters(), lr=learn_rate))
                
        
        elif canc_method=='nn':
            canc = []
            for const in range(np.size(M)):
                #enc.append(Encoder(M[const],modradius[const]))
                dec.append(Decoder(M[const]))
                
                # Adam Optimizer
                optimizer.append([])
                #if const==0:
                #    optimizer=optim.Adam(enc[const].parameters(), lr=learn_rate)
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                if const<lhelp:
                    optimizer[const].append(optim.Adam(enc[const].parameters(),lr=learn_rate*0.01))
                else:
                    enc.append(Encoder(M[const],modradius[const]))
                    optimizer[const].append(optim.Adam(enc[const].parameters(),lr=learn_rate))
                optimizer[const].append(optim.Adam(dec[const].parameters(),lr=learn_rate))
                enc[const].to(device)
                if const>0:
                    canc.append(Canceller(np.product(M)))
                #    optimizer.add_param_group({'params':enc[const].parameters()})
                #    optimizer.add_param_group({'params':canc[const-1].parameters()})
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                    optimizer[const].append(optim.Adam(canc[const-1].parameters(),lr=learn_rate))
            

#optimizer.add_param_group({"params": h_est})

    if canc_method!='nn' and canc_method!='none' and canc_method!='div':
        raise error("Cancellation method invalid. Choose 'none','nn', or 'div'")

    softmax = nn.Softmax(dim=1)

    # Cross Entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # Vary batch size during training
    batch_size_per_epoch = np.linspace(100,10000,num=num_epochs)

    validation_BER = []
    validation_SERs = np.zeros((np.size(M),num_epochs))
    validation_received = []
    if plotting==True:
        decision_region_evolution = []
        constellations = []
        # constellations only used for plotting
        constellation_base = []

        for modulator in range(np.size(M)):
            constellation_base.append([])

    print('Start Training')
    bitnumber = int(np.sum(np.log2(M)))
    gmi = np.zeros(num_epochs)
    gmi_exact = np.zeros((num_epochs, bitnumber))
    for epoch in range(num_epochs):
        batch_labels = torch.empty(int(batch_size_per_epoch[epoch]),np.size(M), device=device)
        validation_BER.append([])
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
                    if canc_method=='none':
                        for dnum in range(np.size(M)):
                            decoded.append(dec[dnum](received))
                    
                    elif canc_method=='div':
                        for dnum in range(np.size(M)):
                            decoded.append([])
                        for dnum in range(np.size(M)):
                            if dnum==0:
                                decoded[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](received))
                                cancelled = torch.view_as_real(torch.view_as_complex(received)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                            else:
                                decoded[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled))
                                cancelled =  torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                    
                    elif canc_method=='nn':
                        for dnum in range(np.size(M)):
                            decoded.append([])
                        for dnum in range(np.size(M)):
                            if dnum==0:
                                decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](received)
                                cancelled =(canc[dnum](received,enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
                            elif dnum==np.size(M)-1:
                                decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled)
                            else:
                                decoded[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled)
                                cancelled=(canc[dnum](cancelled,enc[np.size(M)-dnum-1](softmax(decoded[np.size(M)-dnum-1]))))
            # Calculate Loss
            for num in range(np.size(M)):
            #num=np.mod(epoch,np.size(M))
            # calculate loss as weighted addition of losses for each enc[x] to dec[x] path
                if num==0:
                    loss = weight[0]*loss_fn(decoded[0], batch_labels[:,0].long())
                else:
                    loss += weight[num]*loss_fn(decoded[num], batch_labels[:,num].long())
                    
            loss.backward()
                # compute gradients
                #loss.backward()
            
                # Adapt weights
            for const in range(np.size(M)):
                for elem in optimizer[const]:
                    elem.step()
            
                #optimizer.step()

                # reset gradients
            for const in range(np.size(M)):    
                for elem in optimizer[const]:
                    elem.zero_grad()

            #optimizer.zero_grad()

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
            if num==np.size(M)-1 and canc_method=='none':
                    for dnum in range(np.size(M)):
                        decoded_valid.append(dec[dnum](channel))
            if num==np.size(M)-1 and canc_method=='div':
                for dnum in range(np.size(M)):
                    decoded_valid.append([])
                for dnum in range(np.size(M)):
                    if dnum==0:
                        decoded_valid[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](channel)
                        cancelled = torch.view_as_real(torch.view_as_complex(channel)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))
                    else:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled))
                        cancelled = torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))
            if num==np.size(M)-1 and canc_method=='nn':
                cancelled=[]
                for dnum in range(np.size(M)):
                    decoded_valid.append([])
                for dnum in range(np.size(M)):
                    if dnum==0:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](channel))
                        # canceller
                        cancelled.append(canc[dnum](channel,enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))
                    elif dnum==np.size(M)-1:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled[dnum-1]))
                    else:
                        decoded_valid[np.size(M)-dnum-1]=(dec[np.size(M)-dnum-1](cancelled[dnum-1]))
                        cancelled.append(canc[dnum](cancelled[dnum-1],enc[np.size(M)-dnum-1](softmax(decoded_valid[np.size(M)-dnum-1]))))


        for num in range(len(M)):
            validation_BER[epoch].append(BER(softmax(decoded_valid[num]).detach().cpu().numpy().squeeze(), y_valid[:,num],M[num])[0])
            validation_SERs[num][epoch] = SER(softmax(decoded_valid[num]).detach().cpu().numpy().squeeze(), y_valid[:,num])
            print('Validation BER after epoch %d for encoder %d: ' % (epoch,num) + str(validation_BER[epoch][num].data.tolist()) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))  
            print('Validation SER after epoch %d for encoder %d: %f (loss %1.8f)' % (epoch,num, validation_SERs[num][epoch], loss.detach().cpu().numpy()))              
            if validation_SERs[num][epoch]>0.5 and epoch>10:
                #Weight is increased, when error probability is higher than symbol probability -> misclassification 
                weight[num] += 1
            #elif validation_SERs[num][epoch]<0.01:
            #    optimizer[num][0].param_groups[0]['lr']=0.1*optimizer[num][0].param_groups[0]['lr'] # set encoder learning rate down if converged
        weight=weight/np.sum(weight)*np.size(M) # normalize weight sum
        gmi[epoch],gmi_exact[epoch]=GMI(validation_SERs[:,epoch],M,validation_BER[epoch])
        #print("weight set to "+str(weight))
        print("GMI is: "+ str(gmi[epoch]) + " bit")
        #print("BER is: "+ str(summed_MI[epoch]) + " bit")
        print("SNR is: "+ str(SNR)+" dB")
        if epoch==0:
            enc_best=enc
            dec_best=dec
            if canc_method=='nn':
                canc_best=canc
            best_epoch=0
        elif gmi[epoch]>gmi[best_epoch]:
            enc_best=enc
            dec_best=dec
            if canc_method=='nn':
                canc_best=canc
            best_epoch=0

        if np.sum(validation_SERs[:,epoch])<0.3:
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
                    encoded = torch.reshape(encoded,(M[num+1],int(len(encoded)/M[num+1])))
                else:
                    helper = torch.reshape(constellation_base[num][epoch].repeat(M[num]),(M[num], int(constellation_base[num][epoch].size()[0])))
                    encoded = torch.matmul(torch.transpose(encoded,0,1),helper).flatten()
        
            constellations.append(encoded.detach().cpu().numpy())
        
            # store decision region for generating the animation
            if canc_method=='none':
                for num in range(np.size(M)):
                    decision_region_evolution.append([])
                    mesh_prediction = softmax(dec[num](torch.Tensor(meshgrid).to(device)))
                    decision_region_evolution[num].append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
            elif canc_method=='div':
                for num in range(np.size(M)):
                    decision_region_evolution.append([])
                    if num==0:
                        mesh_prediction = softmax(dec[np.size(M)-num-1](torch.Tensor(meshgrid).to(device)))
                        canc_grid = torch.view_as_real(torch.view_as_complex(torch.Tensor(meshgrid).to(device))/torch.view_as_complex(enc[np.size(M)-num-1](mesh_prediction)))
                    else:
                        mesh_prediction = softmax(dec[np.size(M)-num-1](canc_grid))
                        canc_grid = torch.view_as_real(torch.view_as_complex(canc_grid)/torch.view_as_complex(enc[np.size(M)-num-1](mesh_prediction)))
                    decision_region_evolution[num].append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
            else:
                mesh_prediction=[]
                for dnum in range(np.size(M)):
                    decision_region_evolution.append([])
                    mesh_prediction.append([])
                for dnum in range(np.size(M)):
                    if dnum==0:
                        mesh_prediction[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](torch.Tensor(meshgrid).to(device))
                        cancelled=canc[dnum](torch.Tensor(meshgrid).to(device),enc[np.size(M)-dnum-1](softmax(mesh_prediction[np.size(M)-dnum-1])))
                    elif dnum==np.size(M)-1:
                        mesh_prediction[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled)
                    else:
                        mesh_prediction[np.size(M)-dnum-1]=dec[np.size(M)-dnum-1](cancelled)
                        cancelled=(canc[dnum](cancelled,enc[np.size(M)-dnum-1](softmax(mesh_prediction[np.size(M)-dnum-1]))))
                    decision_region_evolution[np.size(M)-dnum-1].append(0.195*mesh_prediction[np.size(M)-dnum-1].detach().cpu().numpy() +0.4)

            
    print('Training finished')
    if plotting==True:
        plot_training(validation_SERs, validation_received,cvalid,M, constellations, gmi, decision_region_evolution, meshgrid, constellation_base,gmi_exact) 
    max_GMI = np.argmax(gmi)
    if canc_method=='nn':
        return(canc_method,enc_best,dec_best,canc_best, gmi, validation_SERs,gmi_exact)
    else:
        return(canc_method,enc_best,dec_best, gmi, validation_SERs,gmi_exact)

def plot_training(SERs,valid_r,cvalid,M, const, GMIs, decision_region_evolution, meshgrid, constellation_base, gmi_exact):
    cmap = matplotlib.cm.tab20
    base = plt.cm.get_cmap(cmap)
    color_list = base.colors
    new_color_list = [[t/2 + 0.5 for t in color_list[k]] for k in range(len(color_list))]

    sum_SERs = np.sum(SERs, axis=0)/np.size(M)
    min_SER_iter = np.argmin(np.sum(SERs,axis=0))
    max_GMI = np.argmax(GMIs)
    ext_max_plot = 1.2*np.max(np.abs(valid_r[min_SER_iter]))

    plt.figure("SERs",figsize=(3,3))
    font = {#'family': 'serif', 
    'size'   : 10}
    plt.rc('font', **font)
    #plt.rc('text', usetex=True)
    plt.rcParams.update({
    "text.usetex": True,
    #"font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    })

    for num in range(np.size(M)):
        plt.plot(SERs[num],marker='.',linestyle='--', label="Enc"+str(num))
        plt.plot(min_SER_iter,SERs[num][min_SER_iter],marker='o',c='red')
        plt.annotate('Min', (0.95*min_SER_iter,1.4*SERs[num][min_SER_iter]),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('SER')
    plt.grid(which='both')
    plt.legend(loc=1)
    plt.title('SER on Validation Dataset')
    #plt.tight_layout()

    plt.figure("GMIs",figsize=(3,2.5))
    #plt.plot(GMIs,marker='.',linestyle='--',label='Appr.')
    
    for num in range(np.size(gmi_exact[0,:])):
        if num==0:
            t=gmi_exact[:,num]
            plt.fill_between(np.arange(len(t)),t, alpha=0.4)
        else:
            plt.fill_between(np.arange(len(t)),t,t+gmi_exact[:,num],alpha=0.4)
            t+=gmi_exact[:,num]
    plt.plot(t, label='GMI')
    #plt.plot(max_GMI,GMIs[max_GMI],c='red')
    plt.plot(argmax(t),max(t),marker='o',c='red')
    plt.annotate('Max', (0.95*argmax(t),0.9*max(t)),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('GMI')
    #plt.legend(loc=3)
    plt.grid(which='both')
    plt.title('GMI on Validation Dataset')
    plt.tight_layout()

    plt.figure("constellation")
    #plt.subplot(121)
    plt.scatter(np.real(const[min_SER_iter].flatten()),np.imag(const[min_SER_iter].flatten()),c=range(np.product(M)), cmap='tab20',s=50)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid(which='both')
    plt.title('Constellation')

    val_cmplx=valid_r[min_SER_iter][:,0]+1j*valid_r[min_SER_iter][:,1]

    plt.figure("Received signal",figsize=(2.7,2.7))
    #plt.subplot(122)
    plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000], cmap='tab20',s=2)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid()
    plt.title('Received')
    plt.tight_layout()

    print('Minimum mean SER obtained: %1.5f (epoch %d out of %d)' % (sum_SERs[min_SER_iter], min_SER_iter, len(SERs[0])))
    print('Maximum obtained GMI: %1.5f (epoch %d out of %d)' % (GMIs[max_GMI],max_GMI,len(GMIs)))
    print('The corresponding constellation symbols are:\n', const[min_SER_iter])
    
    
    plt.figure("Decision regions", figsize=(5,3))
    for num in range(np.size(M)):
        plt.subplot(1,np.size(M),num+1)
        decision_scatter = np.argmax(decision_region_evolution[num][min_SER_iter], 1)
        if num==0:
            plt.scatter(meshgrid[:,0], meshgrid[:,1], c=decision_scatter,s=3,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:M[num]]))
        else:
            plt.scatter(meshgrid[:,0], meshgrid[:,1], c=decision_scatter,s=3,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[M[num-1]:M[num-1]+M[num]]))
        #plt.scatter(validation_received[min_SER_iter][0:4000,0], validation_received[min_SER_iter][0:4000,1], c=y_valid[0:4000], cmap='tab20',s=4)
        plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000], cmap='tab20',s=2)
        plt.axis('scaled')
        #plt.xlim((-ext_max_plot,ext_max_plot))
        #plt.ylim((-ext_max_plot,ext_max_plot))
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.title('Decoder %d' % num)
        plt.tight_layout()

    plt.figure("Base Constellations")
    for num in range(np.size(M)):
        plt.subplot(1,np.size(M),num+1)
        plt.scatter(np.real(constellation_base[num][min_SER_iter].detach().numpy()),np.imag(constellation_base[num][min_SER_iter].detach().numpy()), c=np.arange(M[num]))
        plt.xlim((-ext_max_plot,ext_max_plot))
        plt.ylim((-ext_max_plot,ext_max_plot))
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.grid()
        plt.tight_layout()



    plt.show()

# ideal modradius: [1,1/3*np.sqrt(2),np.sqrt(2)*1/9]
#canc_method,enc_best,dec_best, smi, validation_SERs=Multipl_NOMA(M=[4,4],sigma_n=[0.01,0.1],train_params=[50,300,0.005],canc_method='none', modradius=[1,1.5/3*np.sqrt(2)], plotting=False)
Multipl_NOMA(M=[4,4],sigma_n=[0.08,0.08],train_params=[60,300,0.0025],canc_method='nn', modradius=[1,1], plotting=True)

#canc_method,enc_best,dec_best,canc_best, smi, validation_SERs=Multipl_NOMA(M=[4,4],sigma_n=[0.01,0.1],train_params=[50,300,0.008],canc_method='nn', modradius=[1,1.5/3*np.sqrt(2)], plotting=False)
#_,en, dec, gmi, ser = Multipl_NOMA([4,4],[0.08,0.08],train_params=[50,300,0.001],canc_method='div', modradius=[1,1], plotting=True)
#Multipl_NOMA(M=[4,4,4],sigma_n=[0.03,0.03,0.02],train_params=[150,1000,0.0008],canc_method='none', modradius=[1,1.5/3*np.sqrt(2),np.sqrt(2)*1.5/9], plotting=True, encoder=enc_best)

