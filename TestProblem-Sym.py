"""
This file uses DMD to solve the symmetric Kornreich and Parsons alpha eigenvalue problem
D. E. KORNREICH and D. KENT PARSONS, “Time– Eigenvalue Calculations in Multi-Region Cartesian Geometry Using Green’s Functions,” 
Ann. Nucl. Energy, 32, 9, 964 (June 2005); https://doi.org/10.1016/j.anucene. 2005.02.004.

It estimates the eigenvalue using Shanks acceleration to check that the code gets the right answer for k = 0.9900716 before doing the alpha calc
"""
import numpy as np
import matplotlib.pyplot as plt
from multigroup_sn import *
import math
from mpmath import *




# In[2]:




def compute_alpha(psi_input,skip,nsteps,I,G,N,dt):
    
    
    it = nsteps-1
    
    #need to reshape matrix
    phi_mat = np.zeros((I*G*N,nsteps))
    for i in range(nsteps):
        phi_mat[:,i] = np.reshape(psi_input[:,:,:,i],I*G*N)
    [u,s,v] = np.linalg.svd(phi_mat[:,skip:it],full_matrices=False)
    print(u.shape,s.shape,v.shape)

    #make diagonal matrix
    #print("Cumulative e-val sum:", (1-np.cumsum(s)/np.sum(s)).tolist())
    spos = s[(1-np.cumsum(s)/np.sum(s)) > 1e-13] #[ np.abs(s) > 1.e-5]
    mat_size = np.min([I*G*N,len(spos)])
    S = np.zeros((mat_size,mat_size))

    unew = 1.0*u[:,0:mat_size]
    vnew = 1.0*v[0:mat_size,:]

    S[np.diag_indices(mat_size)] = 1.0/spos
    print(s)
    Atilde = np.dot(np.dot(np.dot(np.matrix(unew).getH(),phi_mat[:,(skip+1):(it+1)]),np.matrix(vnew).getH()),S)
    print("Atilde size =", Atilde.shape)
    #xnew = np.dot(Atilde,phi_mat[:,0:it])
    #print("Xnew********",xnew[:,1],"phi_mat********",phi_mat[:,1])
    [eigsN,vsN] = np.linalg.eig(Atilde)
    eigsN = (1-1.0/eigsN)/dt
    return eigsN, vsN,u


# In[3]:


def runSlab(cells=100,N=16):
    G = 1
    L = 9
    I = cells*L #540
    hx = L/I
    q = np.ones((I,G))*0
    Xs = np.linspace(hx/2,L-hx/2,I)
    sigma_t = np.ones((I,G))
    nusigma_f = np.zeros((I,G))
    chi = np.ones((I,G))
    sigma_s = np.zeros((I,G,G))
    #first region
    sigma_s[0:I//9,0:G,0:G] = 0.8
    nusigma_f[0:I//9,0:G] = 0.7
    #second region
    sigma_s[I//9:2*I//9,0:G,0:G] = 0.8
    nusigma_f[I//9:2*I//9,0:G] = 0.0
    #third region
    sigma_s[2*I//9:7*I//9,0:G,0:G] = 0.1
    nusigma_f[2*I//9:7*I//9,0:G] = 0.0
    #fourth region
    sigma_s[7*I//9:8*I//9,0:G,0:G] = 0.8
    nusigma_f[7*I//9:8*I//9,0:G] = 0.0
    #fourth region
    sigma_s[8*I//9:I,0:G,0:G] = 0.8
    nusigma_f[8*I//9:I,0:G] = 0.7


#     plt.plot(Xs,chi-np.flip(chi,0))
#     plt.plot(Xs,nusigma_f-np.flip(nusigma_f,0))
#     plt.plot(Xs,sigma_s[:,0]-np.flip(sigma_s[:,0],0))
#     plt.show()
    inv_speed = 1.0

    #N = 196
    MU, W = np.polynomial.legendre.leggauss(N)
    BCs = np.zeros((N,G)) 

    x,k,phi_sol = multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs, 
                          tolerance = 1.0e-8,maxits = 400000, LOUD=1 )
    return x,k,phi_sol




nks = 10
ks = np.zeros(nks)
for i in range(nks):
    x,k,phi_sol = runSlab(50*(i+1),16*(i+1))
    ks[i] = k
    if (i>0):
        T = shanks(ks[:(i+1)])
        for row in T:
            nprint(row)



from mpmath import *
T = shanks(ks)
for row in T:
    nprint(row)




plt.semilogy(x,phi_sol)
plt.show()



G = 1
L = 9
cells = 200
N = 196
I = cells*L #540
hx = L/I
inv_speed = 1.0
MU, W = np.polynomial.legendre.leggauss(N)
BCs = np.zeros((N,G)) 
q = np.ones((I,G))*0
Xs = np.linspace(hx/2,L-hx/2,I)
sigma_t = np.ones((I,G))
nusigma_f = np.zeros((I,G))
chi = np.ones((I,G))
sigma_s = np.zeros((I,G,G))
#first region
sigma_s[0:I//9,0:G,0:G] = 0.8
nusigma_f[0:I//9,0:G] = 0.7
#second region
sigma_s[I//9:2*I//9,0:G,0:G] = 0.8
nusigma_f[I//9:2*I//9,0:G] = 0.0
#third region
sigma_s[2*I//9:7*I//9,0:G,0:G] = 0.1
nusigma_f[2*I//9:7*I//9,0:G] = 0.0
#fourth region
sigma_s[7*I//9:8*I//9,0:G,0:G] = 0.8
nusigma_f[7*I//9:8*I//9,0:G] = 0.0
#fourth region
sigma_s[8*I//9:I,0:G,0:G] = 0.8
nusigma_f[8*I//9:I,0:G] = 0.7
MU, W = np.polynomial.legendre.leggauss(N)
psi0 = np.zeros((I,N,G)) + 1e-12
psi0[0,MU>0,0] = 1
psi0[-1,MU<0,0] = 1
numsteps = 500
dt = 1.0e-1
x,phi,psi = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,numsteps,dt, tolerance = 1.0e-8,maxits = 400000, LOUD=0 )
plt.plot(x,phi[:,0,-1])
plt.show()




print(phi.shape)
plt.plot(x,phi[:,:,-1])
plt.show()




psi.shape
step = 100
included = 395
eigsN, vsN,u = compute_alpha(psi[:,:,:,step:(step+included+1)],0,included,I,G,N,dt)



print(vsN.shape,u.shape)
print(eigsN[ np.abs(np.imag(eigsN)) < 1])





MU, W = np.polynomial.legendre.leggauss(N)
psi0 = np.random.uniform(high=1,low=0,size=(I,N,G)) + 1e-12
numsteps = 500
dt = 1.0e-1
x,phi2,psi2 = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,numsteps,dt, tolerance = 1.0e-8,maxits = 400000, LOUD=0 )
plt.plot(x,phi2[:,0,-1])
plt.show()




print(phi.shape)
plt.plot(x,phi2[:,:,-1])
plt.show()



psi.shape
step = 100
included = 400
eigsN, vsN,u = compute_alpha(psi2[:,:,:,step:(step+included+1)],0,included,I,G,N,dt)




print(vsN.shape,u.shape)
print(eigsN[ np.abs(np.imag(eigsN)) < 1e+0])




plt.plot(x,np.dot(u,vsN[np.argmin(np.abs(eigsN- -0.00616408)),:]))




print(u.shape,vsN.shape)
evect = np.reshape(np.dot(u[:,0:vsN.shape[0]],vsN[:,np.argmin(np.abs(-0.00616408-eigsN))]),(I,N,G))
phi_mat = evect[:,0]*0
print(evect.shape,phi_mat.shape)
for angle in range(N):
    phi_mat +=  evect[:,angle]*W[angle]
    
evect = np.reshape(np.dot(u[:,0:vsN.shape[0]],vsN[:,np.argmin(np.abs(-0.00644849-eigsN))]),(I,N,G))
phi_mat2 = evect[:,0]*0
print(evect.shape,phi_mat.shape)
for angle in range(N):
    phi_mat2 +=  evect[:,angle]*W[angle]


# In[ ]:





# In[ ]:





# In[131]:


eigsN[0]


# In[132]:


np.savez_compressed(file="psi_korn_symm",I=I,psi2=psi2, G=G,N=N,psi = psi,x=x)


# In[174]:


fund=np.loadtxt("brezler.csv",delimiter=",")
fund_sort = np.sort(fund[:,0])
fund_new = fund*0
for i in range(fund_sort.size):
    fund_new[i,:] = fund[np.argmin(np.abs(fund[:,0]-fund_sort[i])),:]
    
sec=np.loadtxt("second_gfm.csv",delimiter=",")
fund_sort = np.sort(sec[:,0])
sec_new = sec*0
for i in range(fund_sort.size):
    sec_new[i,:] = sec[np.argmin(np.abs(sec[:,0]-fund_sort[i])),:]


# In[203]:


print(phi_mat.shape)
plt.plot(x,-np.real(phi_mat)/np.max(np.abs(phi_mat)),label="Fundamental DMD")
#plt.plot(fund_new[:,0],fund_new[:,1]/np.max(np.abs(fund[:,1])),"--")
plt.plot(x,-np.real(phi_mat2)/np.max(np.abs(phi_mat2)),"--",label="Second DMD")
#plt.plot(sec_new[:,0],sec_new[:,1]/np.max(np.abs(sec_new[:,1])),"-.")
plt.legend(loc="best")
plt.show()


# In[201]:


print(phi_mat.shape)
y = np.reshape(-np.real(phi_mat)/np.max(np.abs(phi_mat)),(I))
print(y.shape)
fund_interp = interpolate.interp1d(x,y)
#plt.plot(x,-np.real(phi_mat)/np.max(np.abs(phi_mat)),label="Rightmost DMD")
yinterp = np.reshape(fund_interp(np.reshape(fund_new[:,0],fund_new.shape[0])),fund_new.shape[0])
plt.plot(fund_new[:,0],np.abs(yinterp-fund_new[:,1]/np.max(np.abs(fund[:,1]))),"--")
#plt.plot(x,-np.real(phi_mat2)/np.max(np.abs(phi_mat2)),"--",label="Second DMD")
#plt.plot(sec_new[:,0],sec_new[:,1]/np.max(np.abs(sec_new[:,1])),"-.")
plt.legend(loc="best")
plt.show()


# In[162]:






print(fund[fund[:,0]==fund_sort,:])





