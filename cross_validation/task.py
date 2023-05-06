
# Task 3 choice indication: Difference between using SGD with momentum (as in “train_pt.py”) and Adam optimiser (as
# in "train_tf.py")



import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from torch.nn import CrossEntropyLoss
import numpy as np
from time import time 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print('TASK 3 CHOICE INDICATION: DIFFERENCE BETWEEN USING SGD WITH MOMENTUM (AS IN train_pt.py) AND ADAM OPTIMSER (AS \
    IN train_tf.py' )
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20
    total_epochs = 10
    
    # splitting the trainset into the development set and holdout test set (the latter can be found lower down)
    devset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    dev_len = len(devset) #50000
    trainlen = int(0.8*dev_len) #40000
    vallen = int(0.2*dev_len) #10000
    trainset, valset = torch.utils.data.random_split(devset,[trainlen,vallen])
    trainset2 = list(trainset)

    # 3 fold validation on the using the development set
    num_folds = 3
    foldlen = int(len(trainset)/num_folds) # 
    
    for fold in np.linspace(1,21,num_folds,dtype=int).tolist():
        generator = torch.Generator().manual_seed(fold)

        trainset, valset = torch.utils.data.random_split(devset,[trainlen,vallen],generator=generator)

        # Print data set summary every time the random split is done:
        fold_number =  str(np.linspace(1,21,num_folds,dtype=int).tolist().index(fold))
        cTrainlen = len(trainset)
        cVallen = len(valset)
        print('3 FOLD VALIDATION USING THE DEVSET WITH THE SGD OPTIMISER (UNALTERED)')
        print('Data set summary for random split at fold {}:'.format(fold_number))
        #  print('Index for the split: {}'.format())
        print('Number of elements in trainset: {}'.format(str(cTrainlen)))
        print('Number of elements in valset: {}'.format(str(cVallen)))


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)
        
        ## cnn
        net = Net()

        ## loss and SGD optimiser
        criterion=CrossEntropyLoss()
        optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

        ## train
        for epoch in range(total_epochs):
            
            start_time = time()

            # for trailoader
            train_loss = 0.0
            net.train()

            train_loader_len = len(trainloader)
            val_loader_len = len(valloader)    

            for data, labels in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                
                predictions = net(data)
            
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                # print statistics 
                train_loss += loss.item()

            ##############################################
            # for valloader
            running_val_loss = 0.0
            net.eval()
            # implementing the accuracy metric:

            # initialising accuracy 
            accuracy = 0

            for features, labels in valloader:
                # print(data.shape)
                # print(labels.shape)
                preds = net(features)
                loss = criterion(preds, labels)
                running_val_loss += loss.item()
                ## predicted class
                class_pred_list = []

                for pred in preds:
                    class_prediction = np.argmax(pred.detach().cpu().numpy())
                    class_pred_list.append(class_prediction)
                
                pred_tensor = torch.tensor(class_pred_list)
                all_true = torch.numel(labels) # true positive and true negative
                true_pos = torch.sum(torch.eq(pred_tensor,labels))
                cAccuracy = true_pos/all_true
                accuracy = accuracy + cAccuracy
                    
            accuracy = accuracy * 100

            end_time = time()

            # for each epoch
            print('-----------------------------------------------------------------------')
            print('Fold: {}'.format(str(fold_number)))
            print('Epoch: {} '.format(str(epoch+1))+'train_loss: {} '.format(str(train_loss/train_loader_len)))
            print('val_loss: {} '.format(str(running_val_loss/val_loader_len))+ 'Accuracy: {:.4f}'.format((accuracy/val_loader_len)))
            training_time = end_time-start_time
            print('Training Time (s): {:.4f} '.format(training_time))
            print('-----------------------------------------------------------------------')
    


    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    # THE FOLLOWING CODE IS FOR THE DEVELOPMENT SET WITH THE ADAM OPTIMISER (modification) CROSS VALIDATION
 
    num_folds_adam = 3
    for fold_adam in np.linspace(1,21,num_folds_adam,dtype=int).tolist():
        generator = torch.Generator().manual_seed(fold_adam)

        trainset, valset = torch.utils.data.random_split(devset,[trainlen,vallen],generator=generator)

        # Print data set summary every time the random split is done:
        fold_number_adam =  str(np.linspace(1,21,num_folds,dtype=int).tolist().index(fold_adam))
        cTrainlen = len(trainset)
        cVallen = len(valset)
        print('3 FOLD CROSS VALIDATION ON THE DEVELOPMENT SET WITH THE ADAM OPTIMISER (MODIFIED)')
        print('Data set summary for random split at fold {}:'.format(fold_number_adam))
        #  print('Index for the split: {}'.format())
        print('Number of elements in trainset: {}'.format(str(cTrainlen)))
        print('Number of elements in valset: {}'.format(str(cVallen)))


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)
        
        ## cnn
        net = Net()

        ## loss and Adam optimiser
        criterion=CrossEntropyLoss()
        optimizer=optim.Adam(net.parameters(),lr=0.01)

        ## train
        for epoch in range(total_epochs):
            
            start_time1 = time()

            # for trailoader
            train_loss = 0.0
            net.train()

            train_loader_len = len(trainloader)
            val_loader_len = len(valloader)    

            for data, labels in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                
                predictions = net(data)
            
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                # print statistics 
                train_loss += loss.item()

            ##############################################
            # for valloader
            running_val_loss = 0.0
            net.eval()
            # implementing the accuracy metric:

            # initialising accuracy 
            accuracy = 0
    
            for features, labels in valloader:
                # print(data.shape)
                # print(labels.shape)
                preds = net(features)
                loss = criterion(preds, labels)
                running_val_loss += loss.item()
                ## predicted class
                class_pred_list = []

                for pred in preds:
                    class_prediction = np.argmax(pred.detach().cpu().numpy())
                    class_pred_list.append(class_prediction)
                
                pred_tensor = torch.tensor(class_pred_list)
                all_true = torch.numel(labels) # true positive and true negative
                true_pos = torch.sum(torch.eq(pred_tensor,labels))
                cAccuracy = true_pos/all_true
                accuracy = accuracy + cAccuracy
                    
            accuracy = accuracy * 100

            end_time1 = time()

            # for each epoch
            print('-----------------------------------------------------------------------')
            print('Fold: {}'.format(str(fold_number_adam)))
            print('Epoch: {} '.format(str(epoch+1))+'train_loss: {} '.format(str(train_loss/train_loader_len)))
            print('val_loss: {} '.format(str(running_val_loss/val_loader_len))+ 'Accuracy: {:.4f}'.format((accuracy/val_loader_len)))
            training_time1 = end_time1-start_time1
            print('Training Time (s): {:.4f} '.format(training_time1))
            print('-----------------------------------------------------------------------')

    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print('Train two further models using the entire development set and save the trained models')
    print('-------------------------------------------------------------------------------------')
    print('SGD MODEL TRAINED WITH THE ENTIRE DEVELOPMENT SET - (TO BE SAVED AS MODEL)')

    devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size)
    
    ## cnn
    net = Net()

    ## loss and SGD optimiser
    criterion=CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

    ## train
    for epoch in range(total_epochs):
        

        # for trailoader
        train_loss = 0.0
        net.train()


        for data, labels in devloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            
            predictions = net(data)
        
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            # print statistics 
            train_loss += loss.item()


    # save trained model
    print('model saved as SGD_MODEL_DEVSET.pt')
    torch.save(net.state_dict(), 'SGD_MODEL_DEVSET.pt')



    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print('Train two further models using the entire development set and save the trained models')
    print('-------------------------------------------------------------------------------------')
    print('ADAM MODEL TRAINED WITH THE ENTIRE DEVELOPMENT SET - (TO BE SAVED AS MODEL)')

    devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size)
    
    ## cnn
    net = Net()

    ## loss and ADAM optimiser
    criterion=CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=0.01)

    ## train
    for epoch in range(total_epochs):
        

        # for trailoader
        train_loss = 0.0
        net.train()

        for data, labels in devloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            
            predictions = net(data)
        
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            # print statistics 
            train_loss += loss.item()


    # save trained model
    print('model saved as ADAM_MODEL_DEVSET.pt')
    torch.save(net.state_dict(), 'ADAM_MODEL_DEVSET.pt')

    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print('Report a summary of loss values and metrics on the holdout test set. Compare the results with those obtained during cross-validation.')
    print('-------------------------------------------------------------------------------------')
    print('SUMMARY OF THE LOSS VALUES AND METRICS ON THE HOLDOUT TEST SET USING SGD MODEL')
    

    holdset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    holdloader = torch.utils.data.DataLoader(holdset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataiter = iter(holdloader)
    hold_loader_len = len(holdloader)


    ## load the trained model
    model = Net()
    model.load_state_dict(torch.load('SGD_MODEL_DEVSET.pt'))

    start_time1 = time()
    accuracy = 0
    test_loss = 0
    for features, labels in holdloader:
        # print(data.shape)
        # print(labels.shape)
        preds = model(features)
        loss = criterion(preds, labels)
        test_loss += loss.item()
        ## predicted class
        class_pred_list = []

        for pred in preds:
            class_prediction = np.argmax(pred.detach().cpu().numpy())
            class_pred_list.append(class_prediction)
        
        pred_tensor = torch.tensor(class_pred_list)
        all_true = torch.numel(labels) # true positive and true negative
        true_pos = torch.sum(torch.eq(pred_tensor,labels))
        cAccuracy = true_pos/all_true
        accuracy = accuracy + cAccuracy
            
    accuracy = accuracy * 100

    end_time1 = time()

    # for each epoch
    print('-----------------------------------------------------------------------')
    print('test_loss: {} '.format(str(test_loss/hold_loader_len)))
    print('Accuracy: {:.4f}'.format((accuracy/hold_loader_len)))
    training_time1 = end_time1-start_time1
    print('Training Time (s): {:.4f} '.format(training_time1))
    print('-----------------------------------------------------------------------')



    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print('Report a summary of loss values and metrics on the holdout test set. Compare the results with those obtained during cross-validation.')
    print('-------------------------------------------------------------------------------------')
    print('SUMMARY OF THE LOSS VALUES AND METRICS ON THE HOLDOUT TEST SET USING ADAM MODEL')
    


    ## load the trained model
    model_adam = Net()
    model_adam.load_state_dict(torch.load('ADAM_MODEL_DEVSET.pt'))

    start_time1_adam = time()

    accuracy_adam = 0
    test_loss_adam = 0
    for features, labels in holdloader:
        # print(data.shape)
        # print(labels.shape)
        preds_adam = model_adam(features)
        loss_adam = criterion(preds_adam, labels)
        test_loss_adam += loss_adam.item()
        ## predicted class
        class_pred_list_adam = []

        for pred in preds_adam:
            class_prediction = np.argmax(pred.detach().cpu().numpy())
            class_pred_list_adam.append(class_prediction)
        
        pred_tensor = torch.tensor(class_pred_list_adam)
        all_true = torch.numel(labels) # true positive and true negative
        true_pos = torch.sum(torch.eq(pred_tensor,labels))
        cAccuracy = true_pos/all_true
        accuracy_adam = accuracy_adam + cAccuracy
            
    accuracy_adam = accuracy_adam * 100

    end_time1_adam = time()

    print('-----------------------------------------------------------------------')
    print('test_loss: {} '.format(str(test_loss_adam/hold_loader_len)))
    print('Accuracy: {:.4f}'.format((accuracy_adam/hold_loader_len)))
    training_time1_adam = end_time1_adam-start_time1_adam
    print('Training Time (s): {:.4f} '.format(training_time1_adam))
    print('-----------------------------------------------------------------------')
    print('COMPARE THE RESULTS WITH THOSE OBTAINED DURING CROSS-VALIDATION')
    print('- The accuracy of the SGD and ADAM optimisers is both lower for the holdout test set\
         than for cross validation.')
    print('- Additionally, the accuracy of the sgd optimisation was higher than for the \
        Adame optimiser.')



