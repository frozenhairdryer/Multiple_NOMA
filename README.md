# Master thesis: Design and Analysis of Multiplicative Non-orthogonal Multiple Access Schemes

All code concerning my master thesis at CEL: 

Implemented is a Neural network training a multiplicative modulation format:

received = (mod1 + n_1)*mod2 + n_2

Main Components are:
  * Plotting of the PSD of different pulseshaping functions: sinc, rect, gauss
  * Folder Multipl_NOMA:
    * training_routine has function Multipl_NOMA which is described below
    * training_routine_additive has an equivalent function but uses conventional NOMA, system is: received = (mod1+n_1)+(mod2+n_2)
    * theoreticalMNOMA has the training of a decoder for the theoretical design proposal for a "16 QAM"
    * canc_compare: in this script, the different cancellation methods are run successively and are compared
    * design_eval: compares free learning with the design proposal
    * other files have content which is used in Mulipl_NOMA etc.

## Multipl_NOMA(M=4,sigma_n=0.1,train_params=[50,300,0.005],canc_method='none', modradius=1, plotting=True, encoder=None):

#### M
is a tensor with the number of symbols of each user: example 
    M=torch.tensor([4,4])

#### sigma_n
this tensor defines the noise

#### train_params=[num_epochs,batches_per_epoch, learn_rate]
The training parameters fix the batch sizes, number of training epochs and the learn rate

#### canc_method
canc_method is the chosen cancellation method:
  * division cancellation: canc_method='div'
  * no cancellation: canc_method='none'
  * cancellation with neural network: canc_method='nn'

#### modradius
modradius is the permitted signal amplitude for each encoder
If the design proposal from thesis is not injected, chose modradius=torch.ones(len(M))

M, sigma_n, modradius are lists of the same size

#### plotting=False
This enables/disables the creation of the plots: 
  * Generalized Mutual Information (GMI) during training 
  * Symbol error rates (SERs) of different users during training
  * resulting constellation diagram for best implementation
  * constelation diagrams of different users (constellation_base) for best implementation
  * decoder decision regions

Plots are saved in figures folder in PDF format

#### encoder
trained encoders can be injected in the system. The learning rate will be reduces  by a factor of 0.01 for the injected encoders, as it is assumed that they are already trained.
Injected encoders are added to the end of the list of encoders: They are assumed to be closer to the receiver than new encoders.

example:
    encoder=[enc1, enc2] 

## NN training

### loss function
tbd.
either bitwise loss-function or symbol-wise





