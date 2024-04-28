import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

%matplotlib inline
X_train, y_train = load_data("https://drive.google.com/file/d/1CEql-OEexf9p02M5vCC1RDLXibHYE9Xz/view")
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
def sigmoid(z):

    g = 1 / (1 + np.exp(-z))

    
    return g

value = 0

print (f"sigmoid({value}) = {sigmoid(value)}")
print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)
def compute_cost(X, y, w, b, *argv):

    m,n = X.shape

    loss_sum = 0 

    # Loop over each training example
    for i in range(m): 

        z_wb = 0 
        # Loop over each feature
        for j in range(n): 

            z_wb_ij =  w[j] * X[i][j]
            z_wb += z_wb_ij 
        z_wb += b 

        f_wb = sigmoid(z_wb)
        loss =  -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)# Your code here to calculate loss for a training example

        loss_sum += loss 

    total_cost = (1 / m) * loss_sum      
    

    return total_cost

initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))


# UNIT TESTS
compute_cost_test(compute_cost)
def compute_gradient(X, y, w, b, *argv): 
   m,n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 
    for i in range(m):
              # Calculate f_wb (exactly as you did in the compute_cost function above)
        for k in range(m):   
             # Calculate f_wb (exactly how you did it in the compute_cost function above)
            z_wb = 0
             # Loop over each feature
            for j in range(n): 
                 # Add the corresponding term to z_wb
                z_wb_kj = X[k, j] * w[j]
                z_wb += z_wb_kj

         # Add bias term 
        z_wb += b

         # Calculate the prediction from the model
        f_wb = sigmoid(z_wb) 

              # Calculate the  gradient for b from this example
        dj_db_i = f_wb - y[i]# Your code here to calculate the error

              # add that to dj_db
        dj_db += dj_db_i

              # get dj_dw for each attribute
        for j in range(n):
                  # You code here to calculate the gradient from the i-th example for j-th attribute
                dj_dw_ij =  (f_wb - y[i])* X[i][j]
                dj_dw[j] += dj_dw_ij

          # divide dj_db and dj_dw by total number of examples
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    ### END CODE HERE ###

        
    return dj_db, dj_dw
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test w and b:', dj_db)
print('dj_dw at test w and b:', dj_dw.tolist())

# UNIT TESTS    
compute_gradient_test(compute_gradient)
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
  m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)





plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()



def predict(X, w, b): 
          # number of training examples
        m, n = X.shape   
        p = np.zeros(m)

          ### START CODE HERE ### 
          # Loop over each example
        for i in range(m):   

              # Calculate f_wb (exactly how you did it in the compute_cost function above) 
              # using a couple of lines of code
              for k in range(m):   
                 # Calculate f_wb (exactly how you did it in the compute_cost function above)
                 z_wb = 0
                 # Loop over each feature
                 for j in range(n): 
                     # Add the corresponding term to z_wb
                     z_wb_kj = X[k, j] * w[j]
                     z_wb += z_wb_kj

             # Add bias term 
              z_wb += b

             # Calculate the prediction from the model
              f_wb = sigmoid(z_wb)

              # Calculate the prediction for that training example 
              p[i] = f_wb >= 0.5# Your code here to calculate the prediction based on f_wb

          ### END CODE HERE ### 
        return p




# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

# UNIT TESTS        
predict_test(predict)





#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))




# load dataset
X_train, y_train = load_data("data/ex2data2.txt")


# #### View the variables
# 
# The code below prints the first five values of `X_train` and `y_train` and the type of the variables.
# 

# In[28]:


# print X_train
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))

# print y_train
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))





print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))




# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()



print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)


# Let's also print the first elements of `X_train` and `mapped_X` to see the tranformation.

# In[32]:


print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])



def compute_cost_reg(X, y, w, b, lambda_ = 1):

        m, n = X.shape

          # Calls the compute_cost function that you implemented above
        cost_without_reg = compute_cost(X, y, w, b) 

          # You need to calculate this value
        reg_cost = 0.

          ### START CODE HERE ###
        for j in range(n):
            reg_cost_j = w[j]**2# Your code here to calculate the cost from w[j]
            reg_cost = reg_cost + reg_cost_j
        reg_cost = (lambda_/(2 * m)) * reg_cost
          ### END CODE HERE ### 

          # Add the regularization cost to get the total cost
        total_cost = cost_without_reg + reg_cost

        return total_cost



X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# UNIT TEST    
compute_cost_reg_test(compute_cost_reg)



def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
        m, n = X.shape

        dj_db, dj_dw = compute_gradient(X, y, w, b)

      ### START CODE HERE ###     
      # Loop over the elements of w
        for j in range(n): 

            dj_dw_j_reg =(lambda_ / m) * w[j]  # Your code here to calculate the regularization term for dj_dw[j]

          # Add the regularization term  to the correspoding element of dj_dw
            dj_dw[j] = dj_dw[j] + dj_dw_j_reg

      ### END CODE HERE ###         

        return dj_db, dj_dw



X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1) 
initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
initial_b = 0.5
 
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

# UNIT TESTS    
compute_gradient_reg_test(compute_gradient_reg)




# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01    

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)




plot_decision_boundary(w, b, X_mapped, y_train)
# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()





#Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))


