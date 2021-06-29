from os import error
from numpy.core.fromnumeric import argmax
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
import numpy as cp
import matplotlib.pyplot as plt
import matplotlib
import datetime
import cupy as cp

#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

""" font = {
    'size'   : 10}
    #plt.rc('font', **font)
    #plt.rc('text', usetex=True)
    plt.rcParams.update({
    "text.usetex": True,
    #"font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    }) """

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

class Encoder(nn.Module):
    def __init__(self,M, mradius):
        super(Encoder, self).__init__()
        self.M = torch.as_tensor(M, device='cuda')
        self.mradius = torch.as_tensor(mradius, device='cuda')
        # Define Transmitter Layer: Linear function, M icput neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(self.M,2*self.M, device=device) 
        self.fcT2 = nn.Linear(2*self.M, 2*self.M,device=device)
        self.fcT3 = nn.Linear(2*self.M, 2*self.M,device=device) 
        self.fcT5 = nn.Linear(2*self.M, 2,device=device)
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
        # Define Receiver Layer: Linear function, 2 icput neurons (real and imaginary part), M output neurons (symbols)
        self.M = torch.as_tensor(M, device='cuda')
        self.fcR1 = nn.Linear(2,2*self.M,device=device) 
        self.fcR2 = nn.Linear(2*self.M,2*self.M,device=device) 
        self.fcR3 = nn.Linear(2*self.M,2*self.M,device=device) 
        self.fcR5 = nn.Linear(2*self.M, self.M,device=device) 
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
        self.Mod = torch.as_tensor(Mod, device='cuda')
        self.fcR1 = nn.Linear(2,self.Mod,device=device) 
        self.fcR2 = nn.Linear(2,self.Mod,device=device) 
        self.fcR3 = nn.Linear(self.Mod,self.Mod,device=device) 
        self.fcR4 = nn.Linear(self.Mod,self.Mod,device=device) 
        self.fcR5 = nn.Linear(self.Mod, 2,device=device) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x, decoutput):
        # compute output
        logits = self.cancellation(x, decoutput)
        #norm_factor = torch.max(torch.abs(torch.view_as_complex(logits)).flatten())
        norm_factor = torch.mean(torch.abs(torch.view_as_complex(logits)).flatten()).to(device) # normalize mean amplitude to 1
        logits = logits/norm_factor
        return logits
    
    def cancellation(self,icp, decout):
        out = self.activation_function(self.fcR1(icp)).to(device)
        out += self.activation_function(self.fcR2(decout))
        out = self.activation_function(self.fcR3(out))
        out = self.activation_function(self.fcR4(out))
        logits = self.activation_function(self.fcR5(out))
        return logits
    
def SER(predictions, labels):
    s2 = torch.argmax(predictions, 1)
    return torch.sum( s2!= labels)/predictions.shape[0]

def BER(predictions, labels,m):
    # Bit representation of symbols
    binaries = cp.reshape(cp.unpackbits(cp.arange(0,m,dtype='uint8')), (-1,8))
    binaries = torch.as_tensor(binaries[:,int(8-torch.log2(m)):], device=device)
    y_valid_binary = binaries[labels,:]
    pred_binary = binaries[torch.argmax(predictions, axis=1),:]
    ber=torch.zeros(int(torch.log2(m)), device=device)
    for bit in range(int(torch.log2(m))):
        #print(torch.isclose(pred_binary[:,bit], y_valid_binary[:,bit],rtol=0.5))
        #print(torch.mean(torch.isclose(pred_binary[:,bit], y_valid_binary[:,bit],rtol=0.5), dtype=float))
        ber[bit] = 1-torch.mean(torch.isclose(pred_binary[:,bit], y_valid_binary[:,bit],rtol=0.5), dtype=float)
        if ber[bit]>(0.5): #flip bitmapping
            ber[bit]=1-ber[bit]
    #prob = 
    return ber, y_valid_binary,pred_binary


def GMI(SERs, M, ber=None):
    # gmi estimate or calculation, if bers are given
    gmi_est=0
    for mod in range(len(M)):
        Pe = torch.min(SERs[mod],torch.tensor(0.5)) # all bit are simultaneously wrong
        gmi_est += torch.log2(M[mod])*(1+Pe*torch.log2(Pe+1e-12)+(1-Pe)*torch.log2((1-Pe)+1e-12))
    if ber!=None:
        gmi=torch.zeros(len(M),int(torch.log2(max(M))))
        for num in range(len(M)):
            for x in range(int(torch.log2(M[num]))):
                gmi[num][x]=((1+(ber[num][x]*torch.log2(ber[num][x]+1e-12)+(1-ber[num][x])*torch.log2((1-ber[num][x])+1e-12))))    
        return gmi_est, gmi.flatten()
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
    
    num_epochs=train_params[0]
    batches_per_epoch=train_params[1]
    learn_rate =train_params[2]
    N_valid=10000
    weight=torch.ones(len(M))
    printing=False #suppresses all pinted output but GMI

    # Generate Validation Data
    y_valid = torch.zeros((N_valid,len(M)),dtype=int, device=device)
    for num in range(len(M)):
        y_valid[:,num]= torch.randint(0,M[num],(N_valid,))

    if plotting==True:
        # meshgrid for plotting
        ext_max = 1.5  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
        mgx,mgy = cp.meshgrid(cp.linspace(-ext_max,ext_max,70), cp.linspace(-ext_max,ext_max,70))
        meshgrid = cp.column_stack((cp.reshape(mgx,(-1,1)),cp.reshape(mgy,(-1,1))))
    
    if encoder==None:
        enc=nn.ModuleList().to(device)
        dec=nn.ModuleList().to(device)
        optimizer=[]

        if canc_method=='none' or canc_method=='div':
            for const in range(len(M)):
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
            canc = nn.ModuleList().to(device)
            for const in range(len(M)):
                enc.append(Encoder(M[const],modradius[const]))
                dec.append(Decoder(M[const]))
                enc[const].to(device)
                # Adam Optimizer
                optimizer.append([])
                #if const==0:
                #    optimizer=optim.Adam(enc[const].parameters(), lr=learn_rate)
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                optimizer[const].append(optim.Adam(enc[const].parameters(),lr=float(learn_rate)))
                optimizer[const].append(optim.Adam(dec[const].parameters(),lr=float(learn_rate)))
                if const>0:
                    canc.append(Canceller(torch.prod(M)))
                #    optimizer.add_param_group({'params':enc[const].parameters()})
                #    optimizer.add_param_group({'params':canc[const-1].parameters()})
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                    optimizer[const].append(optim.Adam(canc[const-1].parameters(),lr=float(learn_rate)))
            
    else:
        enc=encoder
        #dec=torch.empty(len(M),device=device)
        dec=nn.ModuleList().to(device)
        optimizer=[]
        lhelp=len(encoder)

        if canc_method=='none' or canc_method=='div':
            for const in range(len(M)):
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
                    optimizer[const].append(optim.Adam(enc[const].parameters(), lr=float(learn_rate)*0.01))
                else:
                    optimizer[const].append(optim.Adam(enc[const].parameters(), lr=float(learn_rate)))
                optimizer[const].append(optim.Adam(dec[const].parameters(), lr=float(learn_rate)))
                
        
        elif canc_method=='nn':
            canc = []
            for const in range(len(M)):
                #enc.append(Encoder(M[const],modradius[const]))
                #dec[const]=(Decoder(M[const]))
                dec.append(Decoder(M[const]))
                
                # Adam Optimizer
                optimizer.append([])
                #if const==0:
                #    optimizer=optim.Adam(enc[const].parameters(), lr=learn_rate)
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                if const<lhelp:
                    optimizer[const].append(optim.Adam(enc[const].parameters(),lr=float(learn_rate)*0.01))
                else:
                    enc.append(Encoder(M[const],modradius[const]))
                    optimizer[const].append(optim.Adam(enc[const].parameters(),lr=float(learn_rate)))
                optimizer[const].append(optim.Adam(dec[const].parameters(),lr=float(learn_rate)))
                enc[const].to(device)
                if const>0:
                    canc.append(Canceller(torch.prod(M)))
                #    optimizer.add_param_group({'params':enc[const].parameters()})
                #    optimizer.add_param_group({'params':canc[const-1].parameters()})
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                    optimizer[const].append(optim.Adam(canc[const-1].parameters(),lr=float(learn_rate)))
            

#optimizer.add_param_group({"params": h_est})

    if canc_method!='nn' and canc_method!='none' and canc_method!='div':
        raise error("Cancellation method invalid. Choose 'none','nn', or 'div'")

    softmax = nn.Softmax(dim=1).to(device)

    # Cross Entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # Vary batch size during training
    batch_size_per_epoch = cp.linspace(100,10000,int(num_epochs))

    validation_BER = []
    validation_SERs = torch.zeros((len(M),int(num_epochs)))
    validation_received = []
    
    print('Start Training')
    bitnumber = int(torch.sum(torch.log2(M)))
    gmi = torch.zeros(int(num_epochs))
    gmi_exact = torch.zeros((int(num_epochs), bitnumber))

    for epoch in range(int(num_epochs)):
        batch_labels = torch.empty(int(batch_size_per_epoch[epoch]),len(M),dtype=torch.long, device=device)
        validation_BER.append([])
        for step in range(int(batches_per_epoch)):
            # Generate training data: In most cases, you have a dataset and do not generate a training dataset during training loop
            # sample new mini-batch directory on the GPU (if available)
            decoded=torch.zeros((int(len(M)),int(batch_size_per_epoch[epoch]),(torch.max(M))), device=device)
            for num in range(len(M)):        
                batch_labels[:,num].random_(int(M[num]))
                batch_labels_o = torch.zeros(int(batch_size_per_epoch[epoch]), int(M[num]), device=device)
                batch_labels_o[range(batch_labels_o.shape[0]), batch_labels[:,num].long()]=1
                batch_labels_onehot = batch_labels_o
                if num==0:
                    # Propagate (training) data through the first transmitter
                    modulated = enc[0](batch_labels_onehot)
                    # Propagate through channel 1
                    received = torch.add(modulated, sigma_n[num]*torch.randn(len(modulated),2).to(device))
                else:
                    modulated = torch.view_as_real(torch.view_as_complex(received)*((torch.view_as_complex(enc[num](batch_labels_onehot)))))
                    received = torch.add(modulated, sigma_n[num]*torch.randn(len(modulated),2).to(device))
                
                if num==len(M)-1:
                    if canc_method=='none':
                        for dnum in range(len(M)):
                            decoded[dnum]=(dec[dnum](received))
                    
                    elif canc_method=='div':
                        for dnum in range(len(M)):
                            if dnum==0:
                                decoded[len(M)-dnum-1]=(dec[len(M)-dnum-1](received))
                                cancelled = torch.view_as_real(torch.view_as_complex(received)/torch.view_as_complex(enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1]))).detach()).to(device)
                            else:
                                decoded[len(M)-dnum-1]=(dec[len(M)-dnum-1](cancelled))
                                cancelled =  torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1]))).detach())
                    
                    elif canc_method=='nn':
                        for dnum in range(len(M)):
                            if dnum==0:
                                decoded[len(M)-dnum-1]=dec[len(M)-dnum-1](received)
                                cancelled =(canc[dnum](received,enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1])).detach()))
                                #cancelled = canc[dnum](received,enc[len(M)-dnum-1](batch_labels_onehot[len(M)-dnum-1]))
                            elif dnum==len(M)-1:
                                decoded[len(M)-dnum-1]=dec[len(M)-dnum-1](cancelled)
                            else:
                                decoded[len(M)-dnum-1]=dec[len(M)-dnum-1](cancelled)
                                cancelled =(canc[dnum](cancelled,enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1])).detach()))
                                #cancelled = canc[dnum](cancelled,enc[len(M)-dnum-1](batch_labels_onehot[len(M)-dnum-1]))
            # Calculate Loss
            for num in range(int(len(M))):
                b, bpred, blabel = BER(softmax(decoded[num]), batch_labels[:,num],M[num])
            #num=cp.mod(epoch,len(M))
            # calculate loss as weighted addition of losses for each enc[x] to dec[x] path
                if num==0:
                    loss = loss_fn(decoded[0], batch_labels[:,0].long())
                else:
                    loss += loss_fn(decoded[num], batch_labels[:,num].long())
                    
            loss.backward()
                # compute gradients
                #loss.backward()
            
                # Adapt weights
            for const in range(len(M)):
                for elem in optimizer[const]:
                    elem.step()
            
                #optimizer.step()

                # reset gradients
            for const in range(len(M)):    
                for elem in optimizer[const]:
                    elem.zero_grad()

            #optimizer.zero_grad()
        with torch.no_grad():
            # compute validation SER, SNR, GMI
            if plotting==True:
                cvalid = torch.zeros(N_valid)
            decoded_valid=torch.zeros((int(len(M)),N_valid,int(torch.max(M))), dtype=torch.float32, device=device)
            SNR = torch.zeros(int(len(M)))
            for num in range(len(M)):
                y_valid_onehot = torch.eye(M[num], device=device)[y_valid[:,num]]
                
                if num==0:
                    encoded = enc[num](y_valid_onehot).to(device)
                    SNR[num] = 20*torch.log10(torch.mean(torch.abs(encoded)))/(float(sigma_n[num]))
                    channel = torch.add(encoded, float(sigma_n[num])*torch.randn(len(encoded),2).to(device))
                    # color map for plot
                    if plotting==True:
                        cvalid=y_valid[:,num]
                else:
                    encoded = torch.view_as_real(torch.view_as_complex(channel)*(torch.view_as_complex(enc[num](y_valid_onehot))))
                    #sigma = sigma_n[num]*cp.mean(cp.abs(torch.view_as_complex(encoded).detach().numpy()))
                    SNR[num] = 20*torch.log10(torch.mean(torch.abs(encoded))/float(sigma_n[num]))
                    channel = torch.add(encoded, float(sigma_n[num])*torch.randn(len(encoded),2).to(device))
                    #color map for plot
                    if plotting==True:
                        cvalid= cvalid+int(M[num])*y_valid[:,num]
                if num==len(M)-1 and canc_method=='none':
                        for dnum in range(len(M)):
                            decoded_valid[dnum]=dec[dnum](channel)
                if num==len(M)-1 and canc_method=='div':
                    for dnum in range(len(M)):
                        if dnum==0:
                            decoded_valid[len(M)-dnum-1]=dec[len(M)-dnum-1](channel)
                            cancelled = torch.view_as_real(torch.view_as_complex(channel)/torch.view_as_complex(enc[len(M)-dnum-1](softmax(decoded_valid[len(M)-dnum-1]))))
                        else:
                            decoded_valid[len(M)-dnum-1]=(dec[len(M)-dnum-1](cancelled))
                            cancelled = torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[len(M)-dnum-1](softmax(decoded_valid[len(M)-dnum-1]))))
                if num==len(M)-1 and canc_method=='nn':
                    #cancelled=[]
                    for dnum in range(len(M)):
                        if dnum==0:
                            decoded_valid[int(len(M))-dnum-1]=dec[int(len(M))-dnum-1](channel).detach()
                            # canceller
                            cancelled = (canc[dnum](channel,enc[int(len(M))-dnum-1](softmax(torch.as_tensor(decoded_valid[len(M)-dnum-1], dtype=torch.float32 ,device=device)))))
                        elif dnum==len(M)-1:
                            decoded_valid[int(len(M))-dnum-1]=dec[int(len(M))-dnum-1](cancelled).detach()
                        else:
                            decoded_valid[len(M)-dnum-1]=dec[len(M)-dnum-1](cancelled)
                            cancelled = (canc[dnum](cancelled[dnum-1],enc[len(M)-dnum-1](softmax(torch.as_tensor(decoded_valid[len(M)-dnum-1], dtype=torch.float32 ,device=device)))))


            for num in range(len(M)):
                validation_BER[epoch].append(BER(softmax(decoded_valid[num]), y_valid[:,num],M[num])[0])
                validation_SERs[num][epoch] = SER(softmax(decoded_valid[num]), y_valid[:,num])
                if printing==True:
                    print('Validation BER after epoch %d for encoder %d: ' % (epoch,num) + str(validation_BER[epoch][num].data.tolist()) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))  
                    print('Validation SER after epoch %d for encoder %d: %f (loss %1.8f)' % (epoch,num, validation_SERs[num][epoch], loss.detach().cpu().numpy()))              
                if validation_SERs[num][epoch]>0.5 and epoch>10:
                    #Weight is increased, when error probability is higher than symbol probability -> misclassification 
                    weight[num] += 1
                #elif validation_SERs[num][epoch]<0.01:
                #    optimizer[num][0].param_groups[0]['lr']=0.1*optimizer[num][0].param_groups[0]['lr'] # set encoder learning rate down if converged
            weight=weight/torch.sum(weight)*len(M) # normalize weight sum
            gmi[epoch],gmi_exact[epoch]=GMI(validation_SERs[:,epoch],M,validation_BER[epoch])
            #print("weight set to "+str(weight))
            print("GMI is: "+ str(torch.sum(gmi_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))
            #print("BER is: "+ str(summed_MI[epoch]) + " bit")
            if printing==True:
                print("SNR is: "+ str(SNR)+" dB")
            if epoch==0:
                enc_best=enc
                dec_best=dec
                if canc_method=='nn':
                    canc_best=canc
                best_epoch=0
            elif torch.sum(gmi_exact[epoch])>torch.sum(gmi_exact[best_epoch]):
                enc_best=enc
                dec_best=dec
                if canc_method=='nn':
                    canc_best=canc
                best_epoch=0

            if torch.sum(validation_SERs[:,epoch])<0.3:
                #set weights back if under threshold
                weight=torch.ones(len(M))
            
            validation_received.append(cp.asarray(channel.detach()))

    if plotting==True:
        decision_region_evolution = []
        #constellations = []
        # constellations only used for plotting
        constellation_base = []
        for num in range(len(M)):
            constellation_base.append(torch.view_as_complex(enc_best[num](torch.eye(int(M[num]), device=device))).cpu().detach().numpy())
        
    
        constellations = cp.asarray(constellation_base[0])
        for num in range(len(M)-1):
            constellationsplus = cp.asarray(constellation_base[num+1])
            constellations = cp.kron(constellations,constellationsplus)
    
    
    # store decision region of best implementation
    if canc_method=='none':
        for num in range(len(M)):
            decision_region_evolution.append([])
            mesh_prediction = softmax(dec_best[num](torch.Tensor(meshgrid).to(device)))
            decision_region_evolution[num].append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
    elif canc_method=='div':
        for num in range(len(M)):
            decision_region_evolution.append([])
            if num==0:
                mesh_prediction = softmax(dec_best[len(M)-num-1](torch.Tensor(meshgrid).to(device)))
                canc_grid = torch.view_as_real(torch.view_as_complex(torch.Tensor(meshgrid).to(device))/torch.view_as_complex(enc[len(M)-num-1](mesh_prediction)))
            else:
                mesh_prediction = softmax(dec_best[len(M)-num-1](canc_grid))
                canc_grid = torch.view_as_real(torch.view_as_complex(canc_grid)/torch.view_as_complex(enc[len(M)-num-1](mesh_prediction)))
            decision_region_evolution[num].append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
    else:
        mesh_prediction=[]
        for dnum in range(len(M)):
            decision_region_evolution.append([])
            mesh_prediction.append([])
        for dnum in range(len(M)):
            if dnum==0:
                mesh_prediction[len(M)-dnum-1]=dec_best[len(M)-dnum-1](torch.Tensor(meshgrid).to(device))
                cancelled=canc_best[dnum](torch.Tensor(meshgrid).to(device),enc_best[len(M)-dnum-1](softmax(mesh_prediction[len(M)-dnum-1])))
            elif dnum==len(M)-1:
                mesh_prediction[len(M)-dnum-1]=dec_best[len(M)-dnum-1](cancelled)
            else:
                mesh_prediction[len(M)-dnum-1]=dec_best[len(M)-dnum-1](cancelled)
                cancelled=(canc_best[dnum](cancelled,enc_best[len(M)-dnum-1](softmax(mesh_prediction[len(M)-dnum-1]))))
            decision_region_evolution[len(M)-dnum-1].append(0.195*mesh_prediction[len(M)-dnum-1].detach().cpu().numpy() +0.4)

            
    print('Training finished')
    if plotting==True:
        plot_training(validation_SERs.cpu().detach().numpy(), cp.asarray(validation_received),cvalid,M, constellations, gmi, decision_region_evolution, meshgrid, constellation_base,gmi_exact) 
    max_GMI = torch.argmax(gmi)
    if canc_method=='nn':
        return(canc_method,enc_best,dec_best,canc_best, gmi, validation_SERs,gmi_exact)
    else:
        return(canc_method,enc_best,dec_best, gmi, validation_SERs,gmi_exact)

def plot_training(SERs,valid_r,cvalid,M, const, GMIs_appr, decision_region_evolution, meshgrid, constellation_base, gmi_exact):
    cmap = matplotlib.cm.tab20
    base = plt.cm.get_cmap(cmap)
    color_list = base.colors
    new_color_list = np.array([[t/2 + 0.49 for t in color_list[k]] for k in range(len(color_list))])

    sum_SERs = np.sum(SERs, axis=0)/len(M)
    min_SER_iter = np.argmin(cp.sum(SERs,axis=0))
    max_GMI = np.argmax(GMIs_appr)
    ext_max_plot = 1.2*np.max(np.abs(valid_r[int(min_SER_iter)]))

    plt.figure("SERs",figsize=(3,3))
    for num in range(len(M)):
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
    plt.plot(GMIs_appr.cpu().detach().numpy(),linestyle='--',label='Appr.')
    #plt.plot(max_GMI,GMIs_appr[max_GMI],c='red')
    for num in range(len(gmi_exact[0,:])):
        if num==0:
            t=gmi_exact[:,num]
            plt.fill_between(np.arange(len(t)),t, alpha=0.4)
        else:
            plt.fill_between(np.arange(len(t)),t,t+gmi_exact[:,num],alpha=0.4)
            t+=gmi_exact[:,num]
    plt.plot(t, label='GMI')
    plt.plot(argmax(t),max(t),marker='o',c='red')
    plt.annotate('Max', (0.95*argmax(t),0.9*max(t)),c='red')
    plt.xlabel('epoch no.')
    plt.ylabel('GMI')
    plt.legend(loc=3)
    plt.grid(which='both')
    plt.title('GMI on Validation Dataset')
    plt.tight_layout()

    constellations = np.array(const.get()).flatten()
    plt.figure("constellation")
    #plt.subplot(121)
    plt.scatter(np.real(constellations),np.imag(constellations),c=range(np.product(M.cpu().detach().numpy())), cmap='tab20',s=50)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid(which='both')
    plt.title('Constellation')

    val_cmplx=np.array((valid_r[min_SER_iter][:,0]+1j*valid_r[min_SER_iter][:,1]).get())

    plt.figure("Received signal",figsize=(2.7,2.7))
    #plt.subplot(122)
    plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000].cpu().detach().numpy(), cmap='tab20',s=2)
    plt.axis('scaled')
    plt.xlabel(r'$\Re\{r\}$')
    plt.ylabel(r'$\Im\{r\}$')
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid()
    plt.title('Received')
    plt.tight_layout()

    print('Minimum mean SER obtained: %1.5f (epoch %d out of %d)' % (sum_SERs[min_SER_iter], min_SER_iter, len(SERs[0])))
    print('Maximum obtained GMI: %1.5f (epoch %d out of %d)' % (GMIs_appr[max_GMI],max_GMI,len(GMIs_appr)))
    print('The corresponding constellation symbols are:\n', const)
    
    
    plt.figure("Decision regions", figsize=(5,3))
    for num in range(len(M)):
        plt.subplot(1,len(M),num+1)
        decision_scatter = np.argmax(decision_region_evolution[num], axis=0)
        grid=meshgrid.get()
        if num==0:
            plt.scatter(grid[:,0], grid[:,1], c=decision_scatter,s=3,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[0:int(M[num])]))
        else:
            plt.scatter(meshgrid[:,0].get(), meshgrid[:,1].get(), c=decision_scatter,s=3,cmap=matplotlib.colors.ListedColormap(colors=new_color_list[int(M[num-1]):int(M[num-1])+int(M[num])]))
        #plt.scatter(validation_received[min_SER_iter][0:4000,0], validation_received[min_SER_iter][0:4000,1], c=y_valid[0:4000], cmap='tab20',s=4)
        plt.scatter(np.real(val_cmplx[0:1000]), np.imag(val_cmplx[0:1000]), c=cvalid[0:1000].cpu().detach().numpy(), cmap='tab20',s=2)
        plt.axis('scaled')
        #plt.xlim((-ext_max_plot,ext_max_plot))
        #plt.ylim((-ext_max_plot,ext_max_plot))
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.title('Decoder %d' % num)
        plt.tight_layout()

    plt.figure("Base Constellations")
    for num in range(len(M)):
        plt.subplot(1,len(M),num+1)
        plt.scatter(np.real(constellation_base[num]),np.imag(constellation_base[num]), c=np.arange(int(M[num])))
        plt.xlim((-ext_max_plot,ext_max_plot))
        plt.ylim((-ext_max_plot,ext_max_plot))
        plt.xlabel(r'$\Re\{r\}$')
        plt.ylabel(r'$\Im\{r\}$')
        plt.grid()
        plt.tight_layout()

    plt.show()

# ideal modradius: [1,1/3*cp.sqrt(2),cp.sqrt(2)*1/9]
# #canc_method,enc_best,dec_best, smi, validation_SERs=Multipl_NOMA(M=[4,4],sigma_n=[0.01,0.1],train_params=[50,300,0.005],canc_method='none', modradius=[1,1.5/3*cp.sqrt(2)], plotting=False)
M=torch.tensor([4,4], dtype=int)
sigma_n=torch.tensor([0.08,0.08], dtype=float)
begin_time = datetime.datetime.now()
Multipl_NOMA(M,sigma_n,train_params=cp.array([60,300,0.002]),canc_method='nn', modradius=cp.array([1,1]), plotting=True)
print(datetime.datetime.now() - begin_time)
#canc_method,enc_best,dec_best,canc_best, smi, validation_SERs=Multipl_NOMA(M=[4,4],sigma_n=[0.01,0.1],train_params=[50,300,0.008],canc_method='nn', modradius=[1,1.5/3*cp.sqrt(2)], plotting=False)
#_,en, dec, gmi, ser = Multipl_NOMA([4,4],[0.08,0.08],train_params=[50,300,0.001],canc_method='div', modradius=[1,1], plotting=True)
#Multipl_NOMA(M=[4,4,4],sigma_n=[0.03,0.03,0.02],train_params=[150,1000,0.0008],canc_method='none', modradius=[1,1.5/3*cp.sqrt(2),cp.sqrt(2)*1.5/9], plotting=True, encoder=enc_best)

