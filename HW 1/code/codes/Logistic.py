import numpy as np

class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)
    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N,d = X.shape
        X_out= np.zeros((N,d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        X_out = np.hstack((np.ones((N,1)), X))
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out  
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        N,d = X.shape 
        
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        y_logis = np.ones((N,1))
        X_train = self.gen_features(X)
        m = len(y_logis)
        print(m)
        for ele in y:
            if y[ele] == -1:
                y_logis[ele] = 0
        loss_calc = 0
        #loss function
        for n in range(N):
            if y_logis[n] == 1:
                loss_calc += -1*np.log(1/(1 + np.exp(-1*np.dot(self.w.T,X_train[n]))))
            else:
                loss_calc += -1*np.log(1-(1/(1 + np.exp(-1*np.dot(self.w.T,X_train[n])))))
        loss = loss_calc/N
        #gradient descent
        thetas = np.zeros((N,2))
        for i in range(N):
            self.w = self.w - (1/(m*10))* np.dot(X_train.T,(1/(1 + np.exp(-1*np.dot(self.w.T,X_train[i]))) - y_logis))
        grad = self.w    
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta, batch_size, num_iters) :
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
            loss_history = []
       
                # ================================================================ #
                # YOUR CODE HERE:
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                # ================================================================ #
            random_samples_indx = np.random.choice(N,batch_size)
            X_batch = X[random_samples_indx,:]
            y_batch = y[random_samples_indx]
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
            loss = 0.0
            grad = np.zeros_like(self.w)
                # ================================================================ #
                # YOUR CODE HERE: 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                # ================================================================ #
           # param = self.reg
            y_pred = self.predict(X_batch)
            #x_new = self.gen_poly_features(X_batch)
              
            sum = 0

            for i in range(self.w.shape[0]):
                for j in range(batch_size):
                    sum += (y[j] - y_pred[j])*X[j][i]
                grad[i] = (self.w[i] - 2*eta*sum - eta*self.w[i])/batch_size
                sum = 0
            self.w = grad
            m = self.m
            loss = (np.sum(np.square(y_pred - y))/batch_size *np.linalg.norm(self.w[1:m]))/2
                
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
            
            loss_history.append(loss)
        return loss_history, self.w
    
    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labelss for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0]+1)
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X_train = self.gen_features(X)
        for i in range(X_train.shape[0]):
            y_hat = 1/(1 + np.exp(-1*np.dot(self.w.T,X_train[i])))
            #print(y_hat)
            if y_hat >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred