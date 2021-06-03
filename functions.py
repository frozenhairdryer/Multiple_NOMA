import torch
import numpy as np

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