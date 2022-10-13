import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import kn
from scipy.integrate import odeint
from scipy.misc import derivative
import time

start = time.time()
print("start time measure")

m_planck = 2.4*10**(18) #GeV reduced Planck mass
dof = 2.0
m_dm1 = 100
m_dm2 = 500

#x = np.linspace(0.1,1000,10000)
x = np.linspace(1.0,5000,500000)

#g_entropy
dof_list = np.loadtxt('/Users/shihyentseng/local/muEDM/eos2020.dat')
dof_entropy_T = interp1d(dof_list[:,0], dof_list[:,1], kind='linear')
dof_entropy_derivative_T = np.empty([len(dof_list),3])
dof_entropy_derivative_T[0,0] = 0.0
dof_entropy_derivative_T[0,2] = (dof_list[0,1]/np.sqrt(dof_list[0,2]))*(1+(dof_list[0,0]/(3*dof_list[0,1]))*0.0)
dof_entropy_derivative_T[-1,0] = dof_list[-1,0]
dof_entropy_derivative_T[-1,2] = (dof_list[-1,1]/np.sqrt(dof_list[-1,2]))*(1+(dof_list[-1,0]/(3*dof_list[-1,1]))*0.0)
for i in range(1, len(dof_list)-1):
    dof_entropy_derivative_T[i,0] = dof_list[i,0]
    dof_entropy_derivative_T[i,1] = derivative(dof_entropy_T, dof_list[i,0], dx=1e-10)
    dof_entropy_derivative_T[i,2] = (dof_list[i,1]/np.sqrt(dof_list[i,2]))*(1+(dof_list[i,0]/(3*dof_list[i,1]))*derivative(dof_entropy_T, dof_list[i,0], dx=1e-10))

dof_list_reverse = dof_list[::-1]
dof_entropy_derivative_T_reverse = dof_entropy_derivative_T[::-1]

dof_list_reverse_new = np.empty([528,4])
for i in range(0, len(dof_list_reverse_new)):
    dof_list_reverse_new[i,0] = m_dm1/dof_list_reverse[i,0]
    dof_list_reverse_new[i,1] = dof_list_reverse[i,1]
    dof_list_reverse_new[i,2] = dof_list_reverse[i,2]
    dof_list_reverse_new[i,3] = dof_entropy_derivative_T_reverse[i,2]

dof_entropy_fun = interp1d(dof_list_reverse_new[:,0], dof_list_reverse_new[:,1], kind='linear')
dof_rho_fun = interp1d(dof_list_reverse_new[:,0], dof_list_reverse_new[:,2], kind='linear')
dof_g_star_fun = interp1d(dof_list_reverse_new[:,0], dof_list_reverse_new[:,3], kind='linear')
x_test = np.linspace(1.0,1.00000000e+03,100000)
plt.xscale('log')
plt.plot(dof_list_reverse_new[:,0], dof_list_reverse_new[:,3], '-')

#cross section
xsv1 = 1.865*10**(-9)
xsv2 = 1.75*10**(-9)

#function of Y_eq
def Y_eq(x,dof):
    Y_eq = 45*dof/(4*np.pi**4*dof_entropy_fun(x))*x**2*kn(2,x)
    return Y_eq

#Boltzmann equation
#xs: cross section, m: DM mass, C: DM asymmetry
def Bol(Y,x,xs,m,C):
    dYdx = xs*(4*np.pi/np.sqrt(90))*m*m_planck*dof_g_star_fun(x)*x**(-2)*(Y_eq(x,dof)**2+C*Y-Y**2)
    return dYdx

#Solve the Boltzmann equation
def Sol(xs,m,C,y0):
    Y = odeint(Bol,y0,x,args=(xs,m,C),hmax=0.01)
    return Y

#Plot the results
plt.figure(figsize=(7.0, 6.0))
xmin, xmax = np.log10(x[0]), np.log10(x[-1])
ymin, ymax = np.log10(1e-17), np.log10(1e-1)
#ymin, ymax = np.log10(1e-1), np.log10(1e2)
plt.xlim(10**(xmin), 10**(xmax))
plt.ylim(10**(ymin), 10**(ymax))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.xticks(fontsize=10, rotation=0)
plt.yticks(fontsize=10, rotation=0)
plt.grid(linestyle='dotted')

plt.plot(x, Sol(xsv1,m_dm1,0,Y_eq(x[0],dof)), color='Red', linestyle='solid', label=r'$\sigma = 1.865\times 10^{-9}$ GeV$^{-2}$, $m_{DM} = 100$ GeV')
#plt.plot(x, Sol(xsv2,m_dm2,0,Y_eq(x[0],dof,dof_entropy)), color='Blue', linestyle='solid', label=r'$\sigma = 1.75\times 10^{-9}$ GeV$^{-2}$, $m_{DM} = 500$ GeV')
plt.plot(x, Y_eq(x,dof), color='black', linestyle='dashed', label=r'$\mathrm{Y}_{\mathrm{eq}}$')
plt.legend()

print('Y0:')
print(Sol(xsv1,m_dm1,0,Y_eq(x[0],dof))[-1])
#print(Sol(xsv2,m_dm2,0,Y_eq(x[0],dof,dof_entropy))[-1])
Omegah1 = 2.755*1e8*m_dm1*Sol(xsv1,m_dm1,0,Y_eq(x[0],dof))[-1]
#Omegah2 = 2.755*1e8*m_dm2*Sol(xsv2,m_dm2,0,Y_eq(x[0],dof,dof_entropy))[-1]
print('Relic abundance')
print(Omegah1)
#print(Omegah2)

end = time.time()
print(end - start)
#plt.savefig("/Users/shihyentseng/local/Dark_matter/DM_test.pdf")
plt.show()