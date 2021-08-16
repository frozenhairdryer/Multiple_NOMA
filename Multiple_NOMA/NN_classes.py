from imports import *

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
        if self.M==1: # dummy user
            return torch.tensor([1,1], dtype=torch.float64,device=device)
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
            modulated = torch.view_as_real((1+self.modradius*torch.view_as_complex(modulated))/(torch.max(torch.abs(1+self.modradius*torch.view_as_complex(modulated)))))
            #todo: fix nomralization of modradius
        return modulated
    

class Decoder(nn.Module):
    def __init__(self,M):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 icput neurons (real and imaginary part), M output neurons (symbols)
        self.M = torch.as_tensor(M, device=device)
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
        out = self.activation_function(self.fcR3(out))
        
        logits = self.activation_function(self.fcR5(out))
        #norm_factor=1
        #if torch.sum(logits)>15:
        #    norm_factor = 15 #norm factor stabilizes training: output higher than 20 leads to nan
        lmax =  torch.max(logits, 1).values
        #for l in range(len(logits)):
        logits = torch.transpose(torch.transpose(logits,0,1) - lmax,0,1) # prevent very high and only low values -> better stability
        return logits+1

class Canceller(nn.Module):
    def __init__(self,Mod):
        super(Canceller, self).__init__()
        self.Mod = torch.as_tensor(Mod, device=device)
        self.fcR1 = nn.Linear(2,self.Mod,device=device) 
        self.fcR2 = nn.Linear(2,self.Mod,device=device) 
        self.fcR3 = nn.Linear(self.Mod,self.Mod,device=device) 
        self.fcR4 = nn.Linear(self.Mod,self.Mod,device=device) 
        self.fcR5 = nn.Linear(self.Mod, 2,device=device) 

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()      

    def forward(self, x, decoutput):
        # compute output
        #x=torch.view_as_real(torch.log(torch.abs(torch.view_as_complex(x))+1e-9)+1j*torch.angle(torch.view_as_complex(x)))
        #decoutput=torch.view_as_real(torch.log(torch.abs(torch.view_as_complex(decoutput))+1e-9)+1j*torch.angle(torch.view_as_complex(decoutput)))
        logits = self.cancellation(x, decoutput)
        #norm_factor = torch.max(torch.abs(torch.view_as_complex(logits)).flatten())
        norm_factor = torch.mean(torch.abs(torch.view_as_complex(logits)).flatten()).to(device) # normalize mean amplitude to 1
        logits = logits/norm_factor
        return logits
    
    def cancellation(self,icp, decout):
        out = self.activation_function(self.fcR1(icp)).to(device)
        out -= self.activation_function(self.fcR2(decout))
        out = self.activation_function(self.fcR3(out))
        #out = self.activation_function(self.fcR4(out))
        logits = self.activation_function(self.fcR5(out))
        #logits = torch.view_as_real(torch.exp(torch.view_as_complex(logits)))
        return logits

def customloss(M, my_outputs, mylabels):
    loss=torch.zeros(int(torch.log2(M)), device=device)
    binaries = torch.tensor(cp.reshape(cp.unpackbits(cp.arange(0,M,dtype='uint8')), (-1,8)),dtype=torch.float32, device=device)
    binaries = binaries[:,int(8-torch.log2(M)):]
    b_labels = binaries[mylabels].int()
    # calculate bitwise estimates
    bitnum = int(torch.log2(M))

    for bit in range(bitnum):
        pos_0 = torch.where(binaries[:,bit]==0)[0]
        pos_1 = torch.where(binaries[:,bit]==1)[0]
        est_0 = torch.sum(torch.index_select(my_outputs,1,pos_0), axis=1)
        est_1 = torch.sum(torch.index_select(my_outputs,1,pos_1), axis=1)
        #print(my_outputs)
        #b_estimates[:,bit]=torch.sum(torch.index_select(P_sym,0,pos_0)*torch.index_select(my_outputs,1,pos_0), axis=1)
        llr = torch.log((est_0+1e-12)/(est_1+1e-12)+1e-12)
        #print(torch.log2(torch.exp((2*b_labels[:,bit]-1)*llr)+1))
        loss[bit]=1/(len(llr))*torch.sum(torch.log2(torch.exp((2*b_labels[:,bit]-1)*llr)+1+1e-12), axis=0)
    return torch.sum(loss)