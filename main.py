import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from p import SPEA2
from p import SPEA2SPECIAL


def paretoValues(function, step=0.1, sym = '-', mini=0, maxi=1,genes=30):
    pareto = np.zeros((int(maxi/step),genes))
    pareto[:,0] = np.arange(mini,maxi,step)
    pareto_sol = function(pareto)
    print(pareto_sol)
    plt.plot(pareto_sol[:,0],pareto_sol[:,1],sym)

def paretoValuesDis(function, sym = '-'):
    pareto = np.ones((32,50))
    c = np.zeros((32,30))
    pareto = np.hstack((c,pareto))
    for i in range(32):
        pareto[i,:i] = 1
    pareto[:,0] = np.arange(0,32,1)

    pareto_sol = function(pareto)
    print(pareto_sol)
    plt.plot(pareto_sol[:,0],pareto_sol[:,1],sym)

def solucionesSPEA(test,color='r',discrete=False, n_genes=30):
    solucion = SPEA2(80,20,250,n_genes,test,discrete=discrete)
    p = test(solucion)
    plt.plot(p[:,0],p[:,1],color+'.')

def solucionSPEAT4(test, color='r'):
    solucion = SPEA2SPECIAL(80,20,250,10,test)
    p = test(solucion)
    plt.plot(p[:,0],p[:,1],color+'.')


def t1(x):
     f1 = x[:,0]
     g = 1 + 9 * (np.sum(x[:,1:], axis=1)/(len(x[0])-1))
     h = 1 - np.sqrt(f1/g)
     f2 = g*h
     return np.array([f1, f2]).T
def t2(x):
     f1 = x[:,0]
     g = 1 + 9 * (np.sum(x[:,1:], axis=1)/(len(x[0])-1))
     h = 1 - (f1/g)**2
     f2 = g*h
     return np.array([f1, f2]).T
def t3(x):
     f1 = x[:,0]
     g = 1 + 9 * (np.sum(x[:,1:], axis=1)/(len(x[0])-1))
     h = 1 - np.sqrt(f1/g)-(f1/g)*np.sin(10*np.pi*f1)
     f2 = g*h
     return np.array([f1, f2]).T

def t4(x):
     f1 = x[:,0]
     g = 1 + 10 * (len(x[0])-1)+ np.sum( (x[:,1:]**2)-10*np.cos(4*np.pi*x[:,1:]), axis=1) 
     h = 1 - np.sqrt(f1/g)
     f2 = g*h
     return np.array([f1, f2]).T

def t5(x):
    x1 = x[:,:30]
    xm = x[:,30:]
    f1 = np.count_nonzero(x1,axis=1)
    f1 = 1 + f1
    uxm = xm.reshape(-1,10,5)#10 x restantes y 5 genes cada uno
    uxm = np.count_nonzero(uxm,axis=2)#Queda una matriz -1 x 10
    vxm1 = uxm < 5
    vxm2 = uxm == 5
    uxm[vxm1] = 2+uxm[vxm1]
    uxm[vxm2] = 1
    vxm = uxm
    g = np.sum(vxm,axis=1)
    print("VALOR G",g)
    h = 1/f1
    f2 = g*h
    return np.array([f1, f2]).T



def t6(x):
     f1 = 1- np.exp(-4*x[:,0])*(np.sin(6*np.pi*x[:,0]))**6
     g = 1 + 9 * (np.sum(x[:,1:], axis=1)/(len(x[0])-1))**0.25
     h = 1 - (f1/g)**2
     f2 = g*h
     return np.array([f1, f2]).T


#solucion = SPEA2(10,5,250,30,t2,maximize=False)
#

#Pruebas
paretoValues(t1)
solucionesSPEA(t1,'b')
plt.show()
#
paretoValues(t2)
solucionesSPEA(t2,'g')
plt.show()
#
paretoValues(t3, 0.001, 'g*')
solucionesSPEA(t3,'k')
plt.show()

paretoValues(t4)
solucionSPEAT4(t4, color='r')
plt.show()


paretoValuesDis(t5)
solucionesSPEA(t5,'r',discrete=True, n_genes = 80)
plt.show()

paretoValues(t6, 0.01)
solucionesSPEA(t6,'y')
plt.show()




#print(p)
#print(solucion)

#plt.plot(p)
#plt.show()

