import numpy as np
import matplotlib.pyplot as plt

from imports import *
from functions import *
from NN_classes import *

class TEncoder(nn.Module):
    def __init__(self,M, mradius):
        super(TEncoder, self).__init__()
        self.M = torch.as_tensor(M, device='cuda')
        self.modradius= mradius 
        if mradius==1:
            self.mapping = 1/np.sqrt(2)*torch.tensor([1+1j,1-1j,-1+1j,-1-1j], device=device)
        else:
            self.mapping = (1+mradius*torch.tensor([1,-1j,1j,-1], device=device))/(1+mradius)

    def forward(self, x):
        modulated = torch.view_as_real(self.mapping[x.long()])
        return modulated

def t_Multipl_NOMA(M=4,sigma_n=0.1,train_params=[50,300,0.005],canc_method='none', modradius=1, plotting=True, encoder=None):
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
    for vnum in range(len(M)):
        y_valid[:,vnum]= torch.randint(0,M[vnum],(N_valid,))

    if plotting==True:
        # meshgrid for plotting
        ext_max = 1.2  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
        mgx,mgy = cp.meshgrid(cp.linspace(-ext_max,ext_max,200), cp.linspace(-ext_max,ext_max,200))
        meshgrid = cp.column_stack((cp.reshape(mgx,(-1,1)),cp.reshape(mgy,(-1,1))))
    
    if encoder==None:
        enc=nn.ModuleList().to(device)
        dec=nn.ModuleList().to(device)
        optimizer=[]

        if canc_method=='none' or canc_method=='div':
            for const in range(len(M)):
                #enc.append(Encoder(M[const], modradius[const]))
                enc.append(TEncoder(M[const],modradius[const]))
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
                #optimizer[const].append(optim.Adam(enc[const].parameters(), lr=float(learn_rate)))
                optimizer[const].append(optim.Adam(dec[const].parameters(), lr=float(learn_rate)))

                
        
        elif canc_method=='nn':
            canc = nn.ModuleList().to(device)
            for const in range(len(M)):
                enc.append(TEncoder(M[const],modradius[const]))
                dec.append(Decoder(M[const]))
                enc[const].to(device)
                # Adam Optimizer
                optimizer.append([])
                #if const==0:
                #    optimizer=optim.Adam(enc[const].parameters(), lr=learn_rate)
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                #optimizer[const].append(optim.Adam(enc[const].parameters(),lr=float(learn_rate)))
                optimizer[const].append(optim.Adam(dec[const].parameters(),lr=float(learn_rate)))
                if const>0:
                    canc.append(Canceller(torch.prod(M)))
                #    optimizer.add_param_group({'params':enc[const].parameters()})
                #    optimizer.add_param_group({'params':canc[const-1].parameters()})
                #    optimizer.add_param_group({'params':dec[const].parameters()})
                    optimizer[const].append(optim.Adam(canc[const-1].parameters(),lr=float(learn_rate)))
        #enc = enc[::-1]
            

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
    gmi = torch.zeros(int(num_epochs),device=device)
    gmi_est2 = torch.zeros(int(num_epochs),device=device)
    gmi_exact = torch.zeros((int(num_epochs), bitnumber), device=device)

    for epoch in range(int(num_epochs)):
        batch_labels = torch.empty(int(batch_size_per_epoch[epoch]),len(M),dtype=torch.long, device=device)
        validation_BER.append([])
        for step in range(int(batches_per_epoch)):
            # Generate training data: In most cases, you have a dataset and do not generate a training dataset during training loop
            # sample new mini-batch directory on the GPU (if available)
            decoded=torch.zeros((int(len(M)),int(batch_size_per_epoch[epoch]),(torch.max(M))), device=device)
            
            for num in range(len(M)):        
                batch_labels[:,num].random_(int(M[num]))
                batch_labels_onehot = torch.zeros(int(batch_size_per_epoch[epoch]), int(M[num]), dtype=torch.long,device=device)
                batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels[:,num].long()]=1
    
                if num==0:
                    # Propagate (training) data through the first transmitter
                    modulated = enc[0](batch_labels[:,num])
                    # Propagate through channel 1
                    received = torch.add(modulated, 0.5*sigma_n[num]*torch.randn(len(modulated),2).to(device))
                else:
                    modulated = torch.view_as_real(torch.view_as_complex(received)*((torch.view_as_complex(enc[num](batch_labels[:,num])))))
                    received = torch.add(modulated, 0.5*sigma_n[num]*torch.randn(len(modulated),2).to(device))
                
                if num==len(M)-1:
                    if canc_method=='none':
                        for dnum in range(len(M)):
                            decoded[dnum]=(dec[dnum](received))
                    
                    elif canc_method=='div':
                        for dnum in range(len(M)):
                            batch_labels_onehot = torch.zeros(int(batch_size_per_epoch[epoch]), int(M[len(M)-dnum-1]), device=device)
                            batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels[:,len(M)-dnum-1].long()]=1
                            if dnum==0:
                                decoded[dnum]=(dec[dnum](received))
                                #genie-aided:
                                cancelled = torch.view_as_real(torch.view_as_complex(received)/torch.view_as_complex(enc[dnum](batch_labels[:,num]).detach()).to(device))
                                #cancelled = torch.view_as_real(torch.view_as_complex(received)/torch.view_as_complex(enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1]))).detach()).to(device)
                            else:
                                decoded[dnum]=(dec[dnum](cancelled))
                                #cancelled =  torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1]))).detach())
                                cancelled =  torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[dnum](batch_labels[:,num]).detach()))
                    
                    elif canc_method=='nn':
                        for dnum in range(len(M)):
                            batch_labels_onehot = torch.zeros(int(batch_size_per_epoch[epoch]), int(M[dnum]), device=device)
                            batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels[:,dnum-1].long()]=1
                            if dnum==0:
                                decoded[dnum]=dec[dnum](received)
                                #cancelled =(canc[dnum](received,enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1])).detach()))
                                cancelled = (canc[dnum](received,enc[dnum](batch_labels[:,dnum]).detach()))
                            elif dnum==len(M)-1:
                                decoded[dnum]=dec[dnum](cancelled)
                            else:
                                decoded[dnum]=dec[dnum](cancelled)
                                #cancelled =(canc[dnum](cancelled,enc[len(M)-dnum-1](softmax(decoded[len(M)-dnum-1])).detach()))
                                cancelled = (canc[dnum](cancelled,enc[dnum](batch_labels[:,dnum]).detach()))
                            #decoded.register_hook(lambda grad: print(grad))
            # Calculate Loss
            for num in range(int(len(M))):
                #b = BER(softmax(decoded[num]), batch_labels[:,num],M[num])
            #num=cp.mod(epoch,len(M))
            # calculate loss as weighted addition of losses for each enc[x] to dec[x] path
                if num==0:
                    #loss = 3*weight[0]*cBCEloss(decoded[0], batch_labels[:,0].long(),M[num])
                    #loss = weight[0]*loss_fn(softmax(decoded[0]), batch_labels[:,0].long())
                    #loss = weight[0]*torch.sum(BER(softmax(decoded[0]), batch_labels[:,0],M[0])[0]).requires_grad()
                    loss = weight[num]*(customloss(M[num],(softmax(decoded[0])), batch_labels[:,0].long()))
                else:
                    #loss = loss.clone()+ 3*weight[num]*cBCEloss(decoded[num], batch_labels[:,num].long(),M[num])
                    #loss = loss.clone() + weight[num]*loss_fn(softmax(decoded[num]), batch_labels[:,num].long())
                    #loss = loss.clone() + weight[num]*torch.sum(BER(softmax(decoded[num]), batch_labels[:,num], M[num])[0])
                    loss = loss.clone() + weight[num]*(customloss(M[num],(softmax(decoded[num])), batch_labels[:,num].long()))

            # compute gradients
            loss.backward()

            # run optimizer
            for const in range(len(M)):
                for elem in optimizer[const]:
                    elem.step()
            

            # reset gradients
            for const in range(len(M)):    
                for elem in optimizer[const]:
                    elem.zero_grad()


        with torch.no_grad(): #no gradient required on validation data 
            # compute validation SER, SNR, GMI
            if plotting==True:
                cvalid = torch.zeros(N_valid)
            decoded_valid=torch.zeros((int(len(M)),N_valid,int(torch.max(M))), dtype=torch.float32, device=device)
            SNR = torch.zeros(int(len(M)), device=device)
            for num in range(len(M)):
                y_valid_onehot = torch.eye(M[num], device=device)[y_valid[:,num]]
                
                if num==0:
                    encoded = enc[num](y_valid[:,num]).to(device)
                    SNR[num] = 20*torch.log10(torch.mean(torch.abs(torch.view_as_complex(encoded)))/float(sigma_n[0]))
                    channel = torch.add(encoded, float(0.5*sigma_n[num])*torch.randn(len(encoded),2).to(device))
                    # color map for plot
                    if plotting==True:
                        cvalid=y_valid[:,num]
                else:
                    encoded = torch.view_as_real(torch.view_as_complex(channel)*(torch.view_as_complex(enc[num](y_valid[:,num]))))
                    SNR[num] = 20*torch.log10(torch.mean(torch.abs(torch.view_as_complex(encoded)))/float(sigma_n[num]))
                    channel = torch.add(encoded, float(0.5*sigma_n[num])*torch.randn(len(encoded),2).to(device))
                    #color map for plot
                    if plotting==True:
                        cvalid= cvalid+int(M[num])*y_valid[:,num]
                if num==len(M)-1 and canc_method=='none':
                        for dnum in range(len(M)):
                            decoded_valid[dnum]=dec[dnum](channel)
                if num==len(M)-1 and canc_method=='div':
                    for dnum in range(len(M)):
                        y_valid_onehot = torch.eye(M[num], device=device)[y_valid[:,len(M)-dnum-1]]
                        if dnum==0:
                            decoded_valid[dnum]=dec[dnum](channel)
                            cancelled = torch.view_as_real(torch.view_as_complex(channel)/torch.view_as_complex(enc[dnum](y_valid[:,dnum])))
                        else:
                            decoded_valid[dnum]=(dec[dnum](cancelled))
                            cancelled = torch.view_as_real(torch.view_as_complex(cancelled)/torch.view_as_complex(enc[dnum](y_valid[:,dnum])))
                if num==len(M)-1 and canc_method=='nn':
                    #cancelled=[]
                    for dnum in range(len(M)):
                        if dnum==0:
                            decoded_valid[dnum]=dec[dnum](channel).detach()
                            # canceller
                            cancelled = (canc[dnum](channel,enc[dnum](y_valid[:,dnum])))
                        elif dnum==len(M)-1:
                            decoded_valid[dnum]=dec[dnum](cancelled).detach()
                        else:
                            decoded_valid[dnum]=dec[dnum](cancelled)
                            cancelled = (canc[dnum](cancelled,enc[dnum](y_valid[:,dnum])))


            for num in range(len(M)):
                validation_BER[epoch].append(BER((softmax(decoded_valid[num])), y_valid[:,num],M[num]))
                validation_SERs[num][epoch] = SER((softmax(decoded_valid[num])), y_valid[:,num])
                if printing==True:
                    print('Validation BER after epoch %d for encoder %d: ' % (epoch,num) + str(validation_BER[epoch][num].data.tolist()) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))  
                    print('Validation SER after epoch %d for encoder %d: %f (loss %1.8f)' % (epoch,num, validation_SERs[num][epoch], loss.detach().cpu().numpy()))              
                # Adapt weights for next training epoch
                if validation_SERs[num][epoch]>0.5 and epoch>10:
                    #Weight is increased, when error probability is higher than symbol probability -> misclassification 
                    weight[num] += 1
                gmi_exact[epoch][num*int(torch.log2(M[num])):(num+1)*int(torch.log2(M[num]))]=GMI(M[num],(softmax(decoded_valid[num])), y_valid[:,num])
            weight=weight/torch.sum(weight)*len(M) # normalize weight sum
            gmi[epoch], t =GMI_est(validation_SERs[:,epoch],M,validation_BER[epoch])
            gmi_est2[epoch] = torch.sum(t)

            print("GMI is: "+ str(torch.sum(gmi_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))
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
        # constellations only used for plotting
        constellation_base = []
        for num in range(len(M)):
            tx = torch.view_as_complex(enc_best[num](torch.linspace(0,int(M[num]-1),int(M[num]))))
            constellation_base.append(tx.cpu().detach().numpy())
        
        constellations = cp.asarray(constellation_base[0])
        for num in range(len(M)-1):
            constellationsplus = cp.asarray(constellation_base[num+1])
            constellations = cp.kron(constellations,constellationsplus)
    
        # store decision region of best implementation
        if canc_method=='none':
            for num in range(len(M)):
                mesh_prediction = (dec_best[num](torch.Tensor(meshgrid).to(device)))
                decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
        elif canc_method=='div':
            for num in range(len(M)):
                if num==0:
                    mesh_prediction = (dec_best[num](torch.Tensor(meshgrid).to(device)))
                    canc_grid = torch.view_as_real(torch.view_as_complex(torch.Tensor(meshgrid).to(device))/torch.view_as_complex(enc[num](torch.argmax(mesh_prediction))))
                else:
                    mesh_prediction = (dec_best[num](canc_grid))
                    canc_grid = torch.view_as_real(torch.view_as_complex(canc_grid)/torch.view_as_complex(enc[num](torch.argmax(mesh_prediction))))
                decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() + 0.4)
        
        
        else:
            for dnum in range(len(M)):
                if dnum==0:
                    mesh_prediction=(dec_best[dnum](torch.Tensor(meshgrid).to(device)))
                    cancelled=canc_best[dnum](torch.Tensor(meshgrid).to(device),enc_best[dnum](torch.argmax(mesh_prediction, axis=1)))
                elif dnum==len(M)-1:
                    mesh_prediction=(dec_best[dnum](cancelled))
                else:
                    mesh_prediction=(dec_best[dnum](cancelled))
                    cancelled=canc_best[dnum](cancelled,enc_best[dnum](torch.argmax(mesh_prediction, axis=1)))
                decision_region_evolution.append(0.195*mesh_prediction.detach().cpu().numpy() +0.4)
        #decision_region_evolution = decision_region_evolution[::-1] 

            
    print('Training finished')
    if plotting==True:
        plot_training(validation_SERs.cpu().detach().numpy(), cp.asarray(validation_received),cvalid,M, constellations, gmi, decision_region_evolution, meshgrid, constellation_base,gmi_exact.detach().cpu().numpy(),gmi_est2.detach().cpu().numpy()) 
    if canc_method=='nn':
        return(canc_method,enc_best,dec_best,canc_best, gmi, validation_SERs,gmi_exact)
    else:
        return(canc_method,enc_best,dec_best, gmi, validation_SERs,gmi_exact)



