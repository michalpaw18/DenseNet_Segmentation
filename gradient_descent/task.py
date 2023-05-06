# task 1 script 

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam
from time import time 
from torchvision import transforms

def polynomial_fun(w,x):
    '''
    w weight vector of size M+1 
    x input scalar variable 
    y function value
    '''
    #
    w=w.type(torch.float32)
    m = torch.numel(w)-1
    y = torch.matmul(w,x**(torch.arange(0,m+1)))

    return y


# Generating the trainining set (100 samples) and the test set (50 samples) for 
# x values in range -20 and 20

# assuming that the training set is 100 pairs and the testing set is 50 pairs
w = torch.tensor([1,2,3,4]).t()
max = 20
min = -20
x_range = max-min
# creating the training set with randomised elements  
x_train = torch.rand(100)*x_range+min
y_train = []
for i in range(len(x_train)):
    y = polynomial_fun(w,x_train[i])
    y_train.append(y)

y_train = torch.tensor(y_train)
# the observed values t are given by the y with the gaussian noise
noise_train = torch.normal(mean=0,std=0.2,size=(100,))
t_train = y_train + noise_train

# for the test set
x_test = torch.rand(50)*x_range+min
y_test = []
for i in range(len(x_test)):
    y = polynomial_fun(w,x_test[i])
    y_test.append(y)

y_test = torch.tensor(y_test)
# the observed values t are given by the y with the gaussian noise
noise_test = torch.normal(mean=0,std=0.2,size=(50,))
t_test = y_test + noise_test



def fit_polynomial_ls(x,t,M):
    '''
    M is the polynomial degree
    x are the data inputs of length N
    t are the targets
    w is the optimised weight vector
    '''

    x = torch.unsqueeze(x,1)
    t = torch.unsqueeze(t,1)
    
    # normalisation (formula: x = (x - x min)/(x max - x min):
    x_min = torch.min(x)
    x_max = torch.max(x)

    x = torch.div((x - x_min),(x_max - x_min))

    t_min = torch.min(t)
    t_max = torch.max(t)

    t = torch.div((t - t_min),(t_max - t_min))


    # transforming the x values in a polynomial matrix (as polynomial regression)
    X = x.pow(torch.arange(0,M+1,1))
    

    # using the normal equation to calculate the w
    w = torch.matmul(torch.inverse(torch.matmul(X.t(),X)),torch.matmul(X.t(),t))
    w = w.t()

    return w



# Using the fit_polynomial_ls with M=4 to compute the weight vector using the training set
M = 4
x = x_train
t = t_train
start_time_ls = time()
w_ls = fit_polynomial_ls(x,t,M)
end_time_ls = time()
print('The optimised weight w using fit_polynomial_ls:\n{}'.format(w_ls),end='\n\n')

time_diff_ls = end_time_ls - start_time_ls
print('\n\n')
print('Time spent in training the ls model {}'.format(time_diff_ls),end='\n\n')




# computing the predicted target values y hat for all x in :

# for the train set using the ls weight:

M=torch.numel(w_ls)-1
x = torch.unsqueeze(x_train,1)
X = x.pow(torch.arange(0,M+1,1))

predictions_train_ls = torch.matmul(X,w_ls.t())
print('Training predicted values using w_ls:\n',predictions_train_ls,end='\n\n')



# Reporting using printed messages the mean (and standard deviation) in difference 
# 
# a) between the observed training data and the underlying “true” polynomial curve; 

mean_obs_true_diff = torch.mean(t_train - y_train)
print('Mean of the difference between the observed training data and the underlying “true” polynomial curve:\n{}'.format(mean_obs_true_diff),end='\n\n')
std_obs_true_diff = torch.std(t_train -y_train)
print('Std of the difference between the observed training data and the underlying “true” polynomial curve:\n{}'.format(std_obs_true_diff),end='\n\n')

# b) between the “LS-predicted” values and the underlying “true” polynomial curve.

mean_lspred_true_diff = torch.mean(predictions_train_ls - y_train)
print('Mean of the difference between the “LS-predicted” values and the underlying “true” polynomial curve:\n{}'.format(mean_lspred_true_diff),end='\n\n')
std_lspred_true_diff = torch.std(predictions_train_ls - y_train)
print('Std of the difference between the “LS-predicted” values and the underlying “true” polynomial curve:\n{}'.format(std_lspred_true_diff),end='\n\n')





# computing the predicted target values y hat for all x in:

# for the test set using ls:

M=torch.numel(w_ls)-1
x = torch.unsqueeze(x_test,1)
X = x.pow(torch.arange(0,M+1,1))

predictions_test_ls = torch.matmul(X,w_ls.t())
print('Testing predicted values using w_ls:\n',predictions_test_ls,end='\n\n')





def model(x,w,b):
    '''
    function used to create the model for the fit_polynomial_sgd using the
    data x, weight w and bias b.
    '''
    prediction = torch.matmul(x,w.t()) + b
    # print(x.shape)
    # print(w.shape)
    return prediction 


# mse loss function

def loss_fun(prediction,label):
    '''
    The function first computes the difference between the predicted and the 
    and label values. It computes the mse error by taking the sum of the squares of the difference and
    divides by the number of elements from the difference.
    '''
    difference = prediction-label
    mse_loss = torch.sum((torch.pow(difference,2)))/difference.numel()
    return mse_loss


def fit_polynomial_sgd(x,t,M,lr,batch_size):
    '''
    x is the input data 
    lr is the learning rate
    batch_size is the mini-batch size
    M is the 1 less than the length of the w
    '''

    x = torch.unsqueeze(x,1)
    t = torch.unsqueeze(t,1)

    # creating the polynomial matrix 
    X = x.pow(torch.arange(0,M+1,1))

    X_cols = X.shape[1]
    t_cols = t.shape[1] # 1 

    # initialising the weights and the bias
    w = torch.randn(t_cols,X_cols,requires_grad=True)
    # w = torch.rand(t_cols,X_cols,requires_grad=True)
    b = torch.randn(t_cols,requires_grad=True)
    # b = torch.rand(t_cols,requires_grad=True)


    dataset_train = TensorDataset(X,t)
    dataloader_train = DataLoader(dataset_train,batch_size,shuffle=True)

    epochs = 100000

    for epoch in range(epochs):
        

        for x, t in dataloader_train:
            predictions = model(x,w,b)
            loss = loss_fun(predictions,t)
            loss.backward()
            
            with torch.no_grad():
                w -= w.grad*lr
                b -= b.grad*lr
                
                # resesting the gradients
                b.grad.zero_()
                w.grad.zero_()


        if (epoch+1) % 10000 == 0:
            print('Epoch number: {} , MSE Loss: {:.2f}'.format(epoch+1,loss.item())) 


    return w




start_time_sgd = time()
M=4
x=x_train
t=t_train
lr=1e-12
batch_size=25

w_sgd = fit_polynomial_sgd(x,t,M,lr,batch_size)
end_time_sgd = time()
print('The optimised weight w using fit_polynomial_sgd:\n{}'.format(w_sgd),end='\n\n')
time_diff_sgd = end_time_sgd - start_time_sgd
print('Time spent in training the sgd model (s) {}'.format(time_diff_sgd),end='\n\n')




# computing the predicted target values y hat for all x in:

# for the train set using sgd:

M=torch.numel(w_sgd)-1
x = torch.unsqueeze(x_train,1)
X = x.pow(torch.arange(0,M+1,1))

predictions_train_sgd = torch.matmul(X,w_sgd.t())
print('Training predicted values using w_sgd:\n',predictions_train_sgd,end='\n\n')



# computing the predicted target values y hat for all x in:

# for the test set using sgd:

M=torch.numel(w_sgd)-1
x = torch.unsqueeze(x_test,1)
X = x.pow(torch.arange(0,M+1,1))

predictions_test_sgd = torch.matmul(X,w_sgd.t())
print('Testing predicted values using w_ls:\n',predictions_test_sgd,end='\n\n')



# Reporting using printed messages the mean (and standard deviation) in difference 
# between the “SGD-predicted” values and the underlying “true” polynomial curve.
mean_sgdpred_true_diff = torch.mean(predictions_train_sgd - y_train)
print('Mean of the difference between the “SGD-predicted” values and the underlying “true” polynomial curve:\n {}'.format(mean_sgdpred_true_diff),end='\n\n')
std_sgdpred_true_diff = torch.std(predictions_train_sgd - y_train)
print('Std of the difference between the “SGD-predicted” values and the underlying “true” polynomial curve:\n {}'.format(std_sgdpred_true_diff),end='\n\n')



# Compare the accuracy of your implementation using the two methods with ground-truth on
# test set and report the root-mean-square-errors (RMSEs) in both 𝐰 and 𝑦 using printed
# messages.
print('Comparing the accuracy of the ls and sgd with ground truth on test set for the prediction and w:')

ls_predictions_rmse = torch.sqrt(loss_fun(predictions_test_ls,y_test))
print('The rmse of the ls prediction with the ground-truth:\n{:.4f}'.format(ls_predictions_rmse))
sgd_predictions_rmse = torch.sqrt(loss_fun(predictions_train_sgd,y_test))
print('The rmse of the sgd prediction with the ground-truth:\n{:.4f}'.format(sgd_predictions_rmse))
print('The rmse of the ls prediction and the rmse of the sgd prediction are quite similar.',end='\n\n')

# ground truth w 
w_ground_truth = torch.tensor([1,2,3,4,0]) 
ls_w_rmse = torch.sqrt(loss_fun(w_ls,w_ground_truth))
print('The rmse of the ls w with the ground-truth:\n{:.4f}'.format(ls_w_rmse))
sgd_w_rmse = torch.sqrt(loss_fun(w_sgd,w_ground_truth)) 
print('The rmse of the sgd w with the ground-truth:\n{:.4f}'.format(sgd_w_rmse))
print('The rmse of the ls w and the rmse of the sgd are quite different.',end='\n\n')



# Compare the speed of the two methods and report time spent in fitting/training (in seconds)
# using printed messages.
print('Time spent in training the ls model (s):\n{:.4f}'.format(time_diff_ls))
print('Time spent in training the sgd model (s):\n{:.4f}'.format(time_diff_sgd))
print('The sgd model takes longer time than the ls model. This is to be expected as \
the sgd model runs for thousands of epochs, whereas the ls model computes the weights \
algebraicly. However, for larger sizes of matrices the ls could potentially take longer \
as the matrix inversion becomes computationaly intensive.',end='\n\n')
