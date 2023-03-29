import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from matplotlib.pyplot import cm
from numpy import size
import imageio
from IPython.display import clear_output


from IPython.display import clear_output
import glob
from IPython.display import Image
import imageio
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython import display





def alpha_m(V):
    return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

def beta_m(V):
    return 4.0*np.exp(-(V+65.0) / 18.0)

def alpha_h(V):
    return 0.07*np.exp(-(V+65.0) / 20.0)

def beta_h(V):
    return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

def alpha_n(V):
    return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

def beta_n(V):
    return 0.125*np.exp(-(V+65) / 80.0)


def tau_m(V):
    return 1/(alpha_m(V)+beta_m(V))
    
def m_inf(V):
    return alpha_m(V)/(alpha_m(V)+beta_m(V))

def tau_n(V):
    return 1/(alpha_n(V)+beta_n(V))
    
def n_inf(V):
    return alpha_n(V)/(alpha_n(V)+beta_n(V))

def tau_h(V):
    return 1/(alpha_h(V)+beta_h(V))
    
def h_inf(V):
    return alpha_h(V)/(alpha_h(V)+beta_h(V))




def I_Na(V, m, h):
    """
    Membrane current (in uA/cm^2)

    |  :param V:
    |  :param m:
    |  :param h:
    |  :return:
    """
    return g_Na * m**3 * h * (V - E_Na)

def I_K(V, n):
    """
    Membrane current (in uA/cm^2)

    |  :param V:
    |  :param h:
    |  :return:
    """
    return g_K  * n**4 * (V - E_K)
#  Leak
def I_L(V):
    """
    Membrane current (in uA/cm^2)

    |  :param V:
    |  :param h:
    |  :return:
    """
    return g_L * (V - E_L)

def I_inj(t):
    """
    External Current

    |  :param t: time
    |  :return: step up to 10 uA/cm^2 at t>100
    |           step down to 0 uA/cm^2 at t>200
    |           step up to 35 uA/cm^2 at t>300
    |           step down to 0 uA/cm^2 at t>400
    """
    return  10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)


def dALLdt(X, t):
    """
    Integrate

    |  :param X:
    |  :param t:
    |  :return: calculate membrane potential & activation variables
    """
    V, m, h, n = X

    dVdt = ( I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V) ) / C_m
    dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return np.array([dVdt, dmdt, dhdt, dndt])
        
def euler(X,t):
    for i in range(1,t.shape[0]):
        X[i,:] = epsilon*dALLdt(X[i-1,:],t[i-1])+X[i-1,:] # Euler Integration Step
    return X



#Model Parameters
C_m  =   1.0
g_Na = 120.0
g_K  =  36.0
g_L  =   0.3
E_Na =  50.0
E_K  = -77.0
E_L  = -54.387

#Simulation Paramaters
epsilon = 0.01
sim_dur = 450.0
t = np.arange(0.0, sim_dur, epsilon)
X = np.zeros((len(t),4))
X[0,0]=-65 #intial V



#Run Simulation
X = euler(X,t)


#Plot

V = X[:,0]
m = X[:,1]
h = X[:,2]
n = X[:,3]
ina = I_Na(V, m, h)
ik = I_K(V, n)
il = I_L(V)

# making a negative current
def I_inj2(t):
  if t > 0 and t < 10 :
    I = -4
  else :
    I = 0


  return  I



def dALLdt(X, t):
    """
    Integrate

    |  :param X:
    |  :param t:
    |  :return: calculate membrane potential & activation variables
    """
    V, m, h, n = X

    dVdt = ( I_inj2(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V) ) / C_m
    dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return np.array([dVdt, dmdt, dhdt, dndt])
        
def euler(X,t):
    for i in range(1,t.shape[0]):
        X[i,:] = epsilon*dALLdt(X[i-1,:],t[i-1])+X[i-1,:] # Euler Integration Step
    return X

#Model Parameters
C_m  =   1.0
g_Na = 120.0
g_K  =  36.0
g_L  =   0.3
E_Na =  50.0
E_K  = -77.0
E_L  = -54.387

#Simulation Paramaters
epsilon = 0.01
sim_dur = 30
t = np.arange(0.0, sim_dur, epsilon)
X = np.zeros((len(t),4))
X[0,0]=-65 #intial V



#Run Simulation
X = euler(X,t)

#Plot

V = X[:,0]
m = X[:,1]
h = X[:,2]
n = X[:,3]
ina = I_Na(V, m, h)
ik = I_K(V, n)
il = I_L(V)

fig,axs = plt.subplots(4,1,figsize=(8,8) , dpi = 120)

axs[0].set_title('Hodgkin-Huxley Neuron')
axs[0].plot(t, V)
axs[0].set_ylabel('V (mV)')
axs[0].set_xlim(0,)


axs[1].plot(t, ina, 'c', label='$I_{Na}$')
axs[1].plot(t, ik, 'y', label='$I_{K}$')
axs[1].plot(t, il, 'm', label='$I_{L}$')
axs[1].set_ylabel('Ionic Current')
axs[1].set_xlim(0,)
axs[1].legend()

axs[2].plot(t, m, 'r', label='m')
axs[2].plot(t, h, 'g', label='h')
axs[2].plot(t, n, 'b', label='n')
axs[2].set_ylabel('Gating Value')
axs[2].set_xlim(0,)
axs[2].legend()

i_inj_values = [I_inj2(t) for t in t]
axs[3].plot(t, i_inj_values, 'k-')
axs[3].set_xlabel('t (ms)')
axs[3].set_ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
axs[3].set_xlim(0,)
clear_output(wait=False)
plt.show()



idx = 0
for ti in range(len(t)) :
  idx = idx + 1 
  print("percent complete = " + str(np.round(idx*100/len(t) , 3)) , end = "\r")
  fig,axs = plt.subplots(4,1,figsize=(8,8) , dpi = 120)

  axs[0].set_title('Hodgkin-Huxley Neuron at time = ' + str(np.round(t[ti], 3)) + "%" ) 
  axs[0].plot(t[0:ti], V[0:ti])
  axs[0].set_ylabel('V (mV)')
  axs[0].set_xlim(0,np.max(t))
  axs[0].set_ylim(-90,40)


  axs[1].plot(t[0:ti], ina[0:ti], 'c', label='$I_{Na}$')
  axs[1].plot(t[0:ti], ik[0:ti], 'y', label='$I_{K}$')
  axs[1].plot(t[0:ti], il[0:ti], 'm', label='$I_{L}$')
  axs[1].set_ylabel('Ionic Current')
  axs[1].set_xlim(0,np.max(t))
  axs[1].set_ylim(-500,500)
  axs[1].legend()

  axs[2].plot(t[0:ti], m[0:ti], 'r', label='m')
  axs[2].plot(t[0:ti], h[0:ti], 'g', label='h')
  axs[2].plot(t[0:ti], n[0:ti], 'b', label='n')
  axs[2].set_ylabel('Gating Value')
  axs[2].set_xlim(0,np.max(t))
  axs[2].set_ylim(0,1.1)
  axs[2].legend()

  i_inj_values = [I_inj2(t) for t in t]
  axs[3].plot(t[0:ti], i_inj_values[0:ti], 'k-')
  axs[3].set_xlabel('t (ms)')
  axs[3].set_ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
  axs[3].set_xlim(0,np.max(t))
  axs[3].set_ylim(-4,1)
  clear_output(wait=False)



  plt.savefig(str(idx) + '.png')
  plt.close('all')
  clear_output(wait=False)


with imageio.get_writer('movie3_afterhyper.gif', mode='I' , duration = 0.02) as writer:
    for i in range(1,np.shape(t)[0]+1):
        filename = str(i) + '.png'
        image = imageio.imread(filename)
        os.remove(filename)
        writer.append_data(image)


with open("movie3_afterhyper.gif",'rb') as f:
    display.Image(data=f.read(), format='png'  )