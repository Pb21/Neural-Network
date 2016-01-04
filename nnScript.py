import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import pow
from math import log
import pickle

labels = np.array([])

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
        
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
        
    return 1.0/(1.0+np.exp(-1.0 * z))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('D:\Machine Learning\mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
     
    
    train_data = np.array([])
    test_data  = np.array([])
    validation_data = np.array([])
    
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    
    validation_label = np.zeros(shape=(1000,1))

    trainx = mat.get('train0')
    test_data = mat.get('test0')
    a = range(trainx.shape[0])
    aperm = np.random.permutation(a)
    
    validation_data = trainx[aperm[0:1000],:]       
    
    train_data = trainx[aperm[1000:],:]        
    train_label = np.zeros(shape=((trainx.shape[0]-1000),1))
    
    test_label = np.zeros(shape=(test_data.shape[0],1))
    
    for i in range(1,10):
       trainx = mat.get('train'+str(i))
       testx = mat.get('test'+str(i))
       a = range(trainx.shape[0])
       aperm = np.random.permutation(a)
       validation_data = np.concatenate(((validation_data,trainx[aperm[0:1000],:])),axis=0)
       b=np.zeros(shape=(1000,1))
       
       b[:] = i 
       validation_label = np.concatenate((validation_label,b),axis=0)
       
       train_data = np.concatenate((train_data,trainx[aperm[1000:],:]),axis=0)       
       c = np.zeros(shape=((trainx.shape[0]-1000),1))
       c[:] = i     
       train_label = np.concatenate((train_label,c),axis=0)
       
       d = np.zeros(shape=((testx.shape[0]),1))
       d[:] = i
       test_label = np.concatenate((test_label,d),axis=0)

       test_data = np.concatenate((test_data,testx),axis=0)
    
    
    train_data = np.double(train_data) 
    test_data = np.double(test_data)
    validation_data = np.double(validation_data)   
    
    train_data /= 255.0
    test_data /= 255.0
    validation_data /= 255.0


    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
#    params = np.linspace(-5,5, num=26)

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    #Your code here
    input_bias = np.ones((training_data.shape[0],1))
    input_data = np.concatenate((training_data,input_bias),axis=1)
    hidden_out =sigmoid((input_data.dot((np.transpose(w1)))))
    
    hidden_bias = np.ones((training_data.shape[0],1))
    outer_in=np.concatenate((hidden_out,hidden_bias),axis=1)
    oil=sigmoid(outer_in.dot(np.transpose(w2)))
    
    ones_mat = np.ones((training_data.shape[0],n_class))
    zeros_mat = np.zeros((training_data.shape[0],n_class))
    
    for i in range(training_label.shape[0]):
        t = training_label[i,0]
        zeros_mat[i,t] = 1
    
    training_label = zeros_mat
      
    obj_val = np.zeros((training_data.shape[0],10))
    obj_val = (training_label*(np.log(oil))+(((ones_mat-training_label)*(np.log(ones_mat-oil)))))
    obj_val = -(obj_val/training_data.shape[0])
            
    w1sqr=w1**2
    w2sqr=w2**2
    wsum = w1sqr.sum() + w2sqr.sum()
    
    obj_val=obj_val.sum()+(lambdaval/(2*training_data.shape[0]))*(wsum)
   
    grad_w2 = np.zeros((n_class,n_hidden+1))
    grad_w2 = (((np.transpose(oil-training_label)).dot(outer_in)) + (lambdaval*w2))/training_data.shape[0]
 
    grad_w1 = np.zeros((n_hidden,n_input+1))    
    hidden_val = (np.ones((training_data.shape[0],n_hidden+1))) - outer_in
  
    grad_w1 = np.dot(np.transpose((np.multiply((np.multiply(hidden_val,outer_in)),(np.dot((oil-training_label),w2))))),input_data)     
    grad_w1 = ((np.delete(grad_w1,n_hidden,0)) + (lambdaval*w1))/training_data.shape[0]
 
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)                
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    training_size = data.shape[0]
    input_bias = np.ones((training_size,1))
    training_data = np.concatenate((data,input_bias),axis=1)       #50000*785

    hidden_in = training_data.dot(np.transpose(w1))                         #50*50000
    hidden_op = sigmoid(hidden_in) 
   
    hidden_bias = np.ones((training_size,1))                                #5000
    hidden_data = np.concatenate((hidden_op,hidden_bias),axis=1)
    
    outer_in = hidden_data.dot(np.transpose(w2))                           #50*50000
    outer_op = sigmoid(outer_in)  
    
    labels = np.argmax(outer_op,axis=1).reshape(training_size,1)
    
    #print labels
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 200;
				   
# set the number of nodes in output unit
n_class = 10;				   



# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 1;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#pickle
#weights = { "w1":w1 , "w2": w2 , "n_hidden":100 , "lambda":0.5 }
#pickle.dump( weights, open( "D:\Machine Learning\params1.pickle", "wb" ) )
fav_color = pickle.load(open("D:\Machine Learning\params1.pickle","rb"))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset
print predicted_label
print "hi"
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
