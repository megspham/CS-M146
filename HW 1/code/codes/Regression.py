import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            X_out = np.zeros((N,2))
            for i in range(N):
                X_out[i,0] = 1
                X_out[i,1] = X[i]
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            for i in range(0, m+1):
                X_out = np.insert(X_out, [i], X**i, axis=1)
                X_out = np.delete(X_out, len(X_out[0])-1, 1)    
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 

        sum = 0
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the linear regression
            # save loss function in loss
            # Calculate the gradient and save it as grad
            #
            # ================================================================ #
            loss = np.sum(np.square(self.predict(X) - y))/N
            for i in range(self.w.shape[0]):
                for j in range(N):
                    sum += (y[j] - self.predict(X)[j])*self.gen_poly_features(X)[j][i]
                grad[i] = (self.w[i] - 2*sum)/N
                sum = 0
            #self.w = grad
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # Calculate the loss function of the polynomial regression with order m
            # ================================================================ #
            loss = np.sum(np.square(self.predict(X) - y))/N
           
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=30, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        self.w = np.zeros_like(self.w)
        for t in np.arange(num_iters):
                X_batch = []
                y_batch = []
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
                param = self.reg
                y_pred = self.predict(X_batch)
                x_new = self.gen_poly_features(X_batch)
              
                sum = 0

                for i in range(self.w.shape[0]):
                    for j in range(batch_size):
                        sum += (y[j] - y_pred[j])*x_new[j][i]
                    grad[i] = (self.w[i] - 2*eta*sum - eta*param*self.w[i])/batch_size
                    sum = 0
                self.w = grad
                m = self.m
                loss = (np.sum(np.square(y_pred - y))/batch_size + param*np.linalg.norm(self.w[1:m]))/2
                
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        return loss_history, self.w
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        loss = 0
        m = self.m
        N,d = X.shape
        x_new = np.matrix(self.gen_poly_features(X))
        y_new = np.array(y).reshape(30, 1)
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # obtain the optimal weights from the closed form solution 
            # ================================================================ #
            self.w = np.matrix.getA(np.matrix.getI(np.matrix.getT(x_new)*x_new)*np.matrix.getT(x_new)*y_new)
            y_pred = self.predict(X)
            loss = np.sum(np.square(y_pred - y))/N
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            self.w = np.matrix.getA(np.matrix.getI(np.matrix.getT(x_new)*x_new)*np.matrix.getT(x_new)*y_new)
            y_pred = self.predict(X)
            loss = np.sum(np.square(y_pred - y))/N
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        N = X.shape[0]
        w = self.w
        x_new = self.gen_poly_features(X)
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            for i in range(N):
                for j in range(w.shape[0]):
                    y_pred[i] += w[j]*x_new[i][j]
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            for i in range(N):
                for j in range(w.shape[0]):
                    y_pred[i] += w[j]*x_new[i][j]           
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return y_pred

