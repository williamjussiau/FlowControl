#!/usr/bin/env python
# coding: utf-8

# <div class="jumbotron text-left"><b>
#     
# This tutorial describes how to use the SMT toolbox to do some Bayesian Optimization (EGO method) to solve unconstrained optimization problem
# <div>
#     
# Rémy Priem and Nathalie BARTOLI ONERA/DTIS/M2CI - April 2020

# <p class="alert alert-success" style="padding:1em">
# To use SMT models, please follow this link : https://github.com/SMTorg/SMT/blob/master/README.md. The documentation is available here: http://smt.readthedocs.io/en/latest/
# </p>
# 
# The reference paper is available 
# here https://www.sciencedirect.com/science/article/pii/S0965997818309360?via%3Dihub 
# 
# or as a preprint: http://mdolab.engin.umich.edu/content/python-surrogate-modeling-framework-derivatives

# <div class="alert alert-info fade in" id="d110">
# <p>In this notebook, two examples are presented to illustrate Bayesian Optimization</p>
# <ol> - a 1D-example (xsinx function) where the algorithm is explicitely given and the use of different criteria is presented </ol>
# <ol> - a 2D-exemple (Rosenbrock function) where the EGO algorithm from SMT is used  </ol>    
# </div>

# # Bayesian Optimization 

# In[1]:


import numpy as np 
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.ion()

def fun(point):
    return np.atleast_2d((point-3.5)*np.sin((point-3.5)/(np.pi)))

X_plot = np.atleast_2d(np.linspace(0, 25, 10000)).T
Y_plot = fun(X_plot)


lines = []
fig = plt.figure(figsize=[5,5])
ax = fig.add_subplot(111)
true_fun, = ax.plot(X_plot,Y_plot)
lines.append(true_fun)
ax.set_title('$x \sin{x}$ function')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#dimension of the problem 
ndim = 1 


# Here, the training data are the points xdata=[0,7,25]. 
# 

# In[2]:


x_data = np.atleast_2d([0,7,25]).T
y_data = fun(x_data)


# Build the GP model with a square exponential kernel with SMT toolbox knowing $(x_{data}, y_{data})$.

# In[3]:


from smt.surrogate_models import KPLS, KRG, KPLSK

########### The Kriging model

# The variable 'theta0' is a list of length ndim.
t = KRG(theta0=[1e-2]*ndim,print_prediction = False, corr='squar_exp')

#Training
t.set_training_values(x_data,y_data)
t.train()

# Prediction of the  points for the plot
Y_GP_plot = t.predict_values(X_plot)
Y_GP_plot_var = t.predict_variances(X_plot)
fig = plt.figure(figsize=[5,5])
ax = fig.add_subplot(111)
true_fun, = ax.plot(X_plot,Y_plot)
data, = ax.plot(x_data,y_data,linestyle='',marker='o')
gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
lines = [true_fun,data,gp,un_gp]
ax.set_title('$x \sin{x}$ function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(lines,['True function','Data','GPR prediction','99 % confidence'])
plt.show()


# 
# Bayesian optimization is defined by Jonas Mockus in (Mockus, 1975) as an optimization technique based upon the minimization of the expected deviation from the extremum of the studied function. 
# 
# The objective function is treated as a black-box function. A Bayesian strategy sees the objective as a random function and places a prior over it. The prior captures our beliefs about the behavior of the function. After gathering the function evaluations, which are treated as data, the prior is updated to form the posterior distribution over the objective function. The posterior distribution, in turn, is used to construct an acquisition function (often also referred to as infill sampling criterion) that determines what the next query point should be.
# 
# One of the earliest bodies of work on Bayesian optimisation that we are aware of is (Kushner, 1962 ; Kushner, 1964). Kushner used Wiener processes for one-dimensional problems. Kushner’s decision model was based on maximizing the probability of improvement, and included a parameter that controlled the trade-off between ‘more global’ and ‘more local’ optimization, in the same spirit as the Exploration/Exploitation trade-off.
# 
# Meanwhile, in the former Soviet Union, Mockus and colleagues developed a multidimensional Bayesian optimization method using linear combinations of Wiener fields, some of which was published in English in (Mockus, 1975). This paper also describes an acquisition function that is based on myopic expected improvement of the posterior, which has been widely adopted in Bayesian optimization as the Expected Improvement function.
# 
# In 1998, Jones used Gaussian processes together with the expected improvement function to successfully perform derivative-free optimization and experimental design through an algorithm called  Efficient  Global  Optimization, or EGO (Jones, 1998).
# 
# ## Efficient Global Optimization
# 
# In what follows, we describe the Efficient Global Optimization (EGO) algorithm, as published in (Jones, 1998).
# 
# Let $F$ be an expensive black-box function to be minimized. We sample $F$ at the different locations  $X = \{x_1, x_2,\ldots,x_n\}$ yielding the responses $Y = \{y_1, y_2,\ldots,y_n\}$. We build a Kriging model (also called Gaussian process) with a mean function $\mu$ and a variance function $\sigma^{2}$.
# 
# The next step is to compute the criterion EI. To do this, let us denote:
# $$f_{min} = \min \{y_1, y_2,\ldots,y_n\}.$$
# The Expected Improvement funtion (EI) can be expressed:
# $$E[I(x)] = E[\max(f_{min}-Y, 0)],$$
# where $Y$ is the random variable following the distribution $\mathcal{N}(\mu(x), \sigma^{2}(x))$.
# By expressing the right-hand side of EI expression as an integral, and applying some tedious integration by parts, one can express the expected improvement in closed form: 
# $$
# E[I(x)] = (f_{min} - \mu(x))\Phi\left(\frac{f_{min} - \mu(x)}{\sigma(x)}\right) + \sigma(x) \phi\left(\frac{f_{min} - \mu(x)}{\sigma(x)}\right)
# $$
# where $\Phi(\cdot)$ and $\phi(\cdot)$ are respectively the cumulative and probability density functions of $\mathcal{N}(0,1)$.
# 
# Next, we determine our next sampling point as :
# \begin{align}
# x_{n+1} = \arg \max_{x} \left(E[I(x)]\right)
# \end{align}
# 
# We then test the response $y_{n+1}$ of our black-box function $F$ at $x_{n+1}$, rebuild the model taking into account the new information gained, and research the point of maximum expected improvement again.
# 
# We summarize here the EGO algorithm:
# 
# EGO(F, $n_{iter}$) \# Find the best minimum of $\operatorname{F}$ in $n_{iter}$ iterations  
# For ($i=0:n_{iter}$)  
# 
# * $mod = {model}(X, Y)$  \# surrogate model based on sample vectors $X$ and $Y$  
# * $f_{min} = \min Y$  
# * $x_{i+1} = \arg \max {EI}(mod, f_{min})$ \# choose $x$ that maximizes EI  
# * $y_{i+1} = {F}(x_{i+1})$ \# Probe the function at most promising point $x_{i+1}$  
# * $X = [X,x_{i+1}]$  
# * $Y = [Y,y_{i+1}]$   
# * $i = i+1$  
# 
# $f_{min} = \min Y$  
# Return : $f_{min}$ \# This is the best known solution after $n_{iter}$ iterations

# Now we want to optimize this function by using Bayesian Optimization and comparing
# - Surrogate Based optimization (SBO)
# - Expected Improvement criterion (EI)
# 

# In a first step we compute the EI criterion

# In[4]:


from scipy.stats import norm
from scipy.optimize import minimize

def EI(GP,points,f_min):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)

    if var.size == 1 and var == 0.0:  # can be use only if one point is computed
        return 0.0
   
    
    ei = args1 + args2
    return ei



Y_GP_plot = t.predict_values(X_plot)
Y_GP_plot_var  =  t.predict_variances(X_plot)
Y_EI_plot = EI(t,X_plot,np.min(y_data))

fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(111)
true_fun, = ax.plot(X_plot,Y_plot)
data, = ax.plot(x_data,y_data,linestyle='',marker='o')
gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
ax1 = ax.twinx()
ei, = ax1.plot(X_plot,Y_EI_plot,color='red')
lines = [true_fun,data,gp,un_gp,ei]
ax.set_title('$x \sin{x}$ function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax1.set_ylabel('ei')
fig.legend(lines,['True function','Data','GPR prediction','99 % confidence','Expected Improvement'],loc=[0.13,0.64])
plt.show()



# Now we compute the EGO method and compare it to other infill criteria 
# - SBO (surrogate based optimization): directly using the prediction of the surrogate model ($\mu$)
# - LCB (Lower Confidence bound): using the confidence interval : $\mu -3 \times \sigma$
# - EI for expected Improvement (EGO)

# In[5]:


#surrogate Based optimization: min the Surrogate model by using the mean mu
def SBO(GP,point):
    res = GP.predict_values(point)
    return res

#lower confidence bound optimization: minimize by using mu - 3*sigma
def LCB(GP,point):
    pred = GP.predict_values(point)
    var = GP.predict_variances(point)
    res = pred-3.*np.sqrt(var)
    return res


# In[6]:


IC = 'EI'


# In[7]:


import matplotlib.image as mpimg
import matplotlib.animation as animation
from IPython.display import HTML


plt.ioff()

x_data = np.atleast_2d([0,7,25]).T
y_data = fun(x_data)

n_iter = 15

gpr = KRG(theta0=[1e-2]*ndim,print_global = False)


for k in range(n_iter):
    x_start = np.atleast_2d(np.random.rand(20)*25).T
    f_min_k = np.min(y_data)
    gpr.set_training_values(x_data,y_data)
    gpr.train()
    if IC == 'EI':
        obj_k = lambda x: -EI(gpr,np.atleast_2d(x),f_min_k)[:,0]
    elif IC =='SBO':
        obj_k = lambda x: SBO(gpr,np.atleast_2d(x))
    elif IC == 'LCB':
        obj_k = lambda x: LCB(gpr,np.atleast_2d(x))
    
    opt_all = np.array([minimize(lambda x: float(obj_k(x)), x_st, method='SLSQP', bounds=[(0,25)]) for x_st in x_start])
    opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]]
    obj_success = np.array([opt_i['fun'] for opt_i in opt_success])
    ind_min = np.argmin(obj_success)
    opt = opt_success[ind_min]
    x_et_k = opt['x']
    
    y_et_k = fun(x_et_k)
    
    y_data = np.atleast_2d(np.append(y_data,y_et_k)).T
    x_data = np.atleast_2d(np.append(x_data,x_et_k)).T
    
    Y_GP_plot = gpr.predict_values(X_plot)
    Y_GP_plot_var  =  gpr.predict_variances(X_plot)
    Y_EI_plot = -EI(gpr,X_plot,f_min_k)

    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(111)
    if IC == 'LCB' or IC == 'SBO':
        ei, = ax.plot(X_plot,Y_EI_plot,color='red')
    else:    
        ax1 = ax.twinx()
        ei, = ax1.plot(X_plot,Y_EI_plot,color='red')
    true_fun, = ax.plot(X_plot,Y_plot)
    data, = ax.plot(x_data[0:k+3],y_data[0:k+3],linestyle='',marker='o',color='orange')
    opt, = ax.plot(x_data[k+3],y_data[k+3],linestyle='',marker='*',color='r')
    gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
    sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
    sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
    un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
    lines = [true_fun,data,gp,un_gp,opt,ei]
    ax.set_title('$x \sin{x}$ function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(lines,['True function','Data','GPR prediction','99 % confidence','Next point to Evaluate','Infill Criteria'])
    plt.savefig('Optimisation %d' %k)
    plt.close(fig)
    
ind_best = np.argmin(y_data)
x_opt = x_data[ind_best]
y_opt = y_data[ind_best]

print('Results : X = %s, Y = %s' %(x_opt,y_opt))

fig = plt.figure(figsize=[10,10])

ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

ims = []
for k in range(n_iter):
    image_pt = mpimg.imread('Optimisation %d.png' %k)
    im = plt.imshow(image_pt)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims,interval=500)
HTML(ani.to_jshtml())



#  ## Use the EGO from SMT 

# In[ ]:





# In[8]:


from smt.applications.ego import EGO
from smt.sampling_methods import LHS


# * Choose your criterion to perform the optimization: EI, SBO or LCB
# * Choose the size of the initial DOE
# * Choose the number of EGO iterations

# ## Try with a 2D function : 2D Rosenbrock function 

# Rosenbrock Function  in dimension N
# 
# $$
# f(\mathbf{x}) = \sum_{i=1}^{N-1} 100 (x_{i+1} - x_i^2 )^2 + (1-x_i)^2 \quad \mbox{where} \quad \mathbf{x} = [x_1, \ldots, x_N] \in \mathbb{R}^N.
# $$
# 
# $$x_i \in [-2,2]$$

# In[9]:


#define the rosenbrock function
def rosenbrock(x):
    """
    Evaluate objective and constraints for the Rosenbrock test case:
    """
    n,dim = x.shape

    #parameters:
    Opt =[]
    Opt_point_scalar = 1
    #construction of O vector
    for i in range(0, dim):
        Opt.append(Opt_point_scalar)

    #Construction of Z vector
    Z= np.zeros((n,dim))
    for i in range(0,dim):
        Z[:,i] = (x[:,i]-Opt[i]+1)

    #Sum
    sum1 = np.zeros((n,1))
    for i in range(0,dim-1):
        sum1[:,0] += 100*(((Z[:,i]**2)-Z[:,i+1])**2)+((Z[:,i]-1)**2)

    return sum1


xlimits=np.array([[-2,2], [-2,2]])


# In[10]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

#To plot the Rosenbrock function
num_plot = 50 #to plot rosenbrock
x = np.linspace(xlimits[0][0],xlimits[0][1],num_plot)
res = []
for x0 in x:
    for x1 in x:
        res.append(rosenbrock(np.array([[x0,x1]])))
res = np.array(res)
res = res.reshape((50,50)).T
X,Y = np.meshgrid(x,x)
fig = plt.figure(figsize=[10,10])
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, res, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.5)
plt.title(' Rosenbrock function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[11]:


criterion='EI' #'EI' or 'SBO' or 'LCB'

#number of points in the initial DOE
ndoe = 10 #(at least ndim+1)

#number of iterations with EGO 
n_iter = 50

#Build the initial DOE, add the random_state option to have the reproducibility of the LHS points
sampling = LHS(xlimits=xlimits, random_state=1)
xdoe = sampling(ndoe)


#EGO call
ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)

x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=rosenbrock)

print('Xopt for Rosenbrock ', x_opt,y_opt, ' obtained using EGO criterion = ', criterion )
print('Check if the optimal point is Xopt= (1,1) with the Y value=0')
print('if not you can increase the number of iterations with n_iter but the CPU will increase also.')
print('---------------------------')


# In[12]:


#To plot the Rosenbrock function
#3D plot
x = np.linspace(xlimits[0][0],xlimits[0][1],num_plot)
res = []
for x0 in x:
    for x1 in x:
        res.append(rosenbrock(np.array([[x0,x1]])))
res = np.array(res)
res = res.reshape((50,50)).T
X,Y = np.meshgrid(x,x)
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, res, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.5)
#to add the points provided by EGO
ax.scatter(x_data[:ndoe,0],x_data[:ndoe,1],y_data[:ndoe],zdir='z',marker = '.',c='k',s=100, label='Initial DOE')
ax.scatter(x_data[ndoe:,0],x_data[ndoe:,1],y_data[ndoe:],zdir='z',marker = 'x',c='r', s=100, label= 'Added point')
ax.scatter(x_opt[0],x_opt[1],y_opt,zdir='z',marker = '*',c='g', s=100, label= 'EGO optimal point')

plt.title(' Rosenbrock function during EGO algorithm')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


#2D plot 
#to add the points provided by EGO
plt.plot(x_data[:ndoe,0],x_data[:ndoe,1],'.', label='Initial DOE')
plt.plot(x_data[ndoe:,0],x_data[ndoe:,1],'x', c='r', label='Added point')
plt.plot(x_opt[:1],x_opt[1:],'*',c='g', label= 'EGO optimal point')
plt.plot([1], [1],'*',c='m', label= 'Optimal point')

plt.title(' Rosenbrock function during EGO algorithm')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


# We can now compare the results by using only the mean information provided by surrogate model approximation
# 

# In[13]:


criterion='SBO' #'EI' or 'SBO' or 'LCB'

#number of points in the initial DOE
ndoe = 10 #(at least ndim+1)

#number of iterations with EGO 
n_iter = 50

#Build the initial DOE
sampling = LHS(xlimits=xlimits, random_state=1)
xdoe = sampling(ndoe)


#EGO call
ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)

x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=rosenbrock)

print('Xopt for Rosenbrock ', x_opt, y_opt, ' obtained using EGO criterion = ', criterion)
print('Check if the optimal point is Xopt=(1,1) with the Y value=0')
print('---------------------------')


# In[ ]:




