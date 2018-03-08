
# coding: utf-8

# In[1]:

import numpy as np
from IPython.display import display, Math, Latex

## basic idea https://github.com/llSourcell/linear_regression_live/blob/master/demo.py


# In[68]:

points = np.genfromtxt("data.csv", delimiter=",")

points[0:2] ## first two data points


# In[5]:

## error for these two data points for m and b = 0
( (31.70700585-(0*32.50234527+0))**2 + (68.77759598-(0*53.42680403+0))**2 ) / 2


# In[6]:

display(Math(r' Error = \frac{1}{N} \sum_{i=1}^{N} (y_{i} - (mx_{i} + b)) ^2'))


# In[46]:

# y = mx + b |||  m is slope, b is y-intercept

def compute_error_for_line_given_points(b, m, points):
    """points are the data"""

    totalError = 0

    for i in range(0, len(points)): # looping through the respective data pairs

        x = points[i, 0]

        y = points[i, 1]

        ## x += 2 means x = x + 2 ##
        totalError += (y - (m * x + b)) ** 2   ### summation part ##
                    # ( squared error for each data pair)    
        
    return totalError / float(len(points)) ## divide by 1/N at the end                   


# In[8]:

compute_error_for_line_given_points(0, 0, points[0:2])


# In[9]:

# derivates of loss function wrt to paramters

display(Math(r' \frac{\partial }{\partial m} = \frac{2}{N} \sum_{i=1}^{N} -x_{i}(y_{i} - (mx_{i} + b))'))

display(Math(r' \frac{\partial }{\partial b} = \frac{2}{N} \sum_{i=1}^{N} -(y_{i} - (mx_{i} + b))'))


# In[10]:

def step_gradient(b_current, m_current, points, learningRate):

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)): # looping through the respective data pairs

        x = points[i, 0]

        y = points[i, 1]

        # updating the gradient (how to update the parameter values in order to minimize overall error)
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current)) # derivates of loss function wrt to b

        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current)) # derivates of loss function wrt to m

    new_b = b_current - (learningRate * b_gradient)  # updating the parameter value for next iteration

    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]


# In[11]:

def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):

    b = initial_b # defined in run()
    m = initial_m # defined in run()

    for i in range(num_iterations):

        b, m = step_gradient(b, m, np.array(points), learning_rate)
        
        print "At iteration {0}, b = {1} and m ={2}".format(i+1, b, m)
        
    return [b, m]


# In[87]:

def run(learning_rate = 0.0001, num_iterations = 1000):

    points = np.genfromtxt("data.csv", delimiter=",")

    initial_b = 0 # initial y-intercept guess

    initial_m = 0 # initial slope guess

    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, 
                                                                              compute_error_for_line_given_points(initial_b, initial_m, points)
                                                                             )
    
    print "Running..."

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, 
                                                                      compute_error_for_line_given_points(b, m, points)
                                                                     )


# In[13]:

run()


# In[14]:

run(num_iterations = 5)


# # Replacing loops via numpy computations

# In[67]:

def compute_error_for_line_given_points_np(b, m, points):
    """points are the data"""
   
    x = np.array(points[:,0]) 
    y = np.array(points[:,1]) 
    
    return np.sum((y - (m * x + b)) ** 2) / len(points) ## divide by 1/N at the end      


print ('error difference for the two methods:')
compute_error_for_line_given_points_np(0, 0, points) - compute_error_for_line_given_points(0, 0, points)


# # run time comparison 

# In[66]:

get_ipython().magic(u'timeit compute_error_for_line_given_points_np(0, 0, points)')


# In[55]:

get_ipython().magic(u'timeit compute_error_for_line_given_points(0, 0, points)')


# In[94]:

def step_gradient_np(b_current, m_current, points, learningRate):

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)): # looping through the respective data pairs

        x = points[i, 0]

        y = points[i, 1]

        # updating the gradient (how to update the parameter values in order to minimize overall error)
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current)) # derivates of loss function wrt to b

        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current)) # derivates of loss function wrt to m

    new_b = b_current - (learningRate * b_gradient)  # updating the parameter value for next iteration

    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]


# In[80]:

def gradient_descent_runner_np(points, initial_b, initial_m, learning_rate, num_iterations):

    b = initial_b # defined in run()
    m = initial_m # defined in run()

    for i in range(num_iterations):

        b, m = step_gradient_np(b, m, np.array(points), learning_rate)
        
        #print "At iteration {0}, b = {1} and m ={2}".format(i+1, b, m)
        
    return [b, m]


# In[81]:

def run_with_np_method(learning_rate = 0.0001, num_iterations = 1000):

    points = np.genfromtxt("data.csv", delimiter=",")

    initial_b = 0 # initial y-intercept guess

    initial_m = 0 # initial slope guess

    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, 
                                                                              compute_error_for_line_given_points_np(initial_b, initial_m, points)
                                                                             )
    
    print "Running..."

    [b, m] = gradient_descent_runner_np(points, initial_b, initial_m, learning_rate, num_iterations)

    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, 
                                                                      compute_error_for_line_given_points_np(b, m, points)
                                                                     )


# In[88]:

get_ipython().magic(u'timeit run()')


# In[102]:

get_ipython().magic(u'timeit run_with_np_method()')


# In[101]:

run_with_np_method()


# In[100]:

def step_gradient_np(b_current, m_current, points, learningRate):

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    x = np.array(points[:,0]) 
    y = np.array(points[:,1]) 

        # updating the gradient (how to update the parameter values in order to minimize overall error)
    b_gradient = np.sum(-(2/N) * (y - ((m_current * x) + b_current))) # derivates of loss function wrt to b

    m_gradient = np.sum(-(2/N) * x * (y - ((m_current * x) + b_current))) # derivates of loss function wrt to m

    new_b = b_current - (learningRate * b_gradient)  # updating the parameter value for next iteration

    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]


# In[98]:

b_gradient = 0
m_gradient = 0
N = float(len(points))

x = np.array(points[:,0]) 
y = np.array(points[:,1]) 

        # updating the gradient (how to update the parameter values in order to minimize overall error)
b_gradient += -(2/N) * (y - ((0 * x) + 0)) # derivates of loss function wrt to b
new_b = 0 - (0.00001 * b_gradient)  # updating the parameter value for next iteration


# In[99]:

b_gradient


# In[ ]:



