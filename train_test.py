import torch
from models import RMSLELoss, Net
import numpy as np
import matplotlib.pyplot as plt
def train_nn_reg(inputsize, train_data_loader, epochs, lr):
    """train using Net neural network in model for the regression problem 
       and print the loss after each iteration
       return model with list of loss after each iteration"""
    
    
    model = Net(inputsize = inputsize).cuda()
    criterion = RMSLELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
    losses = []
    for epoch in range(epochs):
        running_loss = 0
        for data in train_data_loader:
            inputs = data[0].cuda().float() 
            labels = data[1].cuda().float()
            optimizer.zero_grad()
            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = criterion(outputs, labels.reshape(labels.shape[0],1))
            running_loss = running_loss + loss.item()
            # get gradients w.r.t to parameters
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # update parameters
            optimizer.step()
        running_loss = running_loss / len(train_data_loader)
        losses.append(np.sqrt(running_loss))
        print('epoch {}, loss {}'.format(epoch, np.sqrt(running_loss)))
        
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    return model, losses 


def test_nn_reg(model, test_data_loader):
    criterion = RMSLELoss()
    rn_loss = 0
    for data in test_data_loader:
        inputs = data[0].cuda().float() 
        labels = data[1].cuda().float()
        loss = criterion(model(inputs),labels)
        rn_loss = rn_loss + loss
    rn_loss = rn_loss / len(test_data_loader)
    print(f"the testing RMLE: {rn_loss}")
    