import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from models import RMSLELoss, Net
def clean_data(train_set_panda):
    """
    Take a panda dataframe and convert all categorical values to numerical
      - fill nan values with the mean values in numerical columns 
      - drop 4 columns due to nan:
    """
    
    for clm in train_set_panda.columns:
        # check if it is a categorical values
        if train_set_panda[clm].dtype.name == 'object':
            cat_vals = train_set_panda[clm].unique()
            numerical_vals = [i for i in range(1,len(cat_vals)+1)]
            # replace categorical variable with numerical 
            train_set_panda[clm] = train_set_panda[clm].replace(cat_vals,numerical_vals )

        else:
            train_set_panda[clm] = train_set_panda[clm].fillna(train_set_panda['LotFrontage'].mean())



    # removing Alley, PoolQC, Fence, MiscFeature because of the nan values
    train_set_panda.drop('Alley', inplace=True, axis=1)
    train_set_panda.drop('PoolQC', inplace=True, axis=1)
    train_set_panda.drop('Fence', inplace=True, axis=1)
    train_set_panda.drop('MiscFeature', inplace=True, axis=1)
    train_set_panda['LotFrontage'] = train_set_panda['LotFrontage'].fillna(train_set_panda['LotFrontage'].mean())
    train_set_panda['GarageYrBlt'] = train_set_panda['GarageYrBlt'].fillna(train_set_panda['GarageYrBlt'].mean())
    
    return train_set_panda


def abnormal(train_set_panda, test_size, contamination):
    
    """
    take panda dataframe that contains X and y and split them into numpy arrays 
        - do the split based on the test size
        - apply abnormal detection based on contamination
        - return X_train, y_train, X_test, y_test
   
    """
    
    
    X_train = np.array(train_set_panda)[:,0:np.array(train_set_panda).shape[1]-1]
    Y_train = np.array(train_set_panda)[:,np.array(train_set_panda).shape[1]-1]

    # split the training set into testing and training 
    X_train, X_test, y_train, y_test = train_test_split(X_train,Y_train, test_size = test_size)

    iso = IsolationForest(contamination = contamination)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]
    
    return X_train, X_test, y_train, y_test


def visualize_pca(X_train,X_test):
    """
    convert the data into two dimension and visualize in a scatter plot for training and testing without 
    the y values
    """
    pca_train = PCA(n_components = 2)
    pca_train.fit(X_train)
    plt.scatter(pca_train.transform(X_train)[:,0], pca_train.transform(X_train)[:,1])

    # testing dataset (orange)
    pca_test = PCA(n_components = 2)
    pca_test.fit(X_test)
    plt.scatter(pca_test.transform(X_test)[:,0], pca_test.transform(X_test)[:,1])
    
    plt.xlabel("X pca")
    plt.ylabel("Y pca")
    plt.legend(["Training Dat", "Testing Data"])
    
def prep_dataloader(X_train,X_test, y_train, y_test, batch_size):
    # preprocessing for pytorch as a tensor
    prep_train_set = []
    for i in range(X_train.shape[0]):
        prep_train_set.append((X_train[i,:], y_train[i]))
    prep_test_set = []
    for i in range(X_test.shape[0]):
        prep_test_set.append((X_test[i,:], y_test[i]))
    train_data_loader = torch.utils.data.DataLoader(prep_train_set, batch_size= batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(prep_test_set, batch_size= batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader
