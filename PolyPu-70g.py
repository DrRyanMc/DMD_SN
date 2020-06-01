#!/usr/bin/python3
"""
This code produces the results in section V.B of McClarren, 'Calculating Time Eigenvalues of the Neutron Transport Equation with Dynamic Mode Decomposition', NSE 2018
It can take a long time to run and it is expecting to be run interactively as it is set up
"""
import numpy as np
import matplotlib.pyplot as plt
from multigroup_sn import *
#from multigroup_sn import multigroup_k
import math


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

#load in 70g data
metal_tmp = np.load("metal_data.npz")
metal_scat = metal_tmp['metal_scat']
metal_sig_t = metal_tmp['metal_sig_t']
metal_chi = metal_tmp['metal_chi']
metal_nu_sig_f = metal_tmp['metal_nu_sig_f']



poly_tmp = np.load("poly_data.npz")
poly_scat = poly_tmp['poly_scat']
poly_sig_t = poly_tmp['poly_sig_t']
poly_chi = poly_tmp['poly_chi']
poly_nu_sig_f = poly_tmp['poly_nu_sig_f']

inv_speed = metal_tmp['metal_inv_speed']
group_edges = metal_tmp['metal_group_edges']

group_des = -np.diff(group_edges)
group_centers = (group_edges[0:-1] + group_edges[1:])*0.5

inner_thick = 20
I = 30 #300
L = 25.25
Lx = L
hx = L/I
ref_thick = 3 #reflector thickness
#k = 0.995432853326 with ref_thick = 8.5 and L = 11 and I = 200
G = group_centers.size
q = np.ones((I,G))*0
Xs = np.linspace(hx/2,L-hx/2,I)
sigma_t = np.zeros((I,G))
nusigma_f = np.zeros((I,G))
chi = np.zeros((I,G))
sigma_s = np.zeros((I,G,G))
sigma_t[:,0:G] = metal_sig_t
sigma_s[:,0:G,0:G] = metal_scat.transpose()
chi[:,0:G] = metal_chi
nusigma_f[:,0:G] = metal_nu_sig_f
for i in range(I):
    
    if ((Xs[i]+hx/2 ) <=  Lx/2 + inner_thick/2) and (Xs[i] - hx/2 >= Lx/2- inner_thick/2):
        sigma_t[i,0:G] = poly_sig_t
        sigma_s[i,0:G,0:G] = poly_scat.transpose()
        nusigma_f[i,0:G] = poly_nu_sig_f
        chi[i,0:G] = poly_chi
    
    elif (np.abs(Xs[i]+hx/2) <= (ref_thick)/2) or (Xs[i]-hx/2 >= Lx-(ref_thick)/2):
        sigma_t[i,0:G] = poly_sig_t
        sigma_s[i,0:G,0:G] = poly_scat.transpose()
        chi[i,0:G] = poly_chi
        nusigma_f[i,0:G] = poly_nu_sig_f
N = 4
MU, W = np.polynomial.legendre.leggauss(N)
BCs = np.zeros((N,G))
BCs[(N//2):N,:] = 0.0

x,k,phi_sol,phi_thermal, phi_epithermal,phi_fast = multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,group_edges, 
                          tolerance = 1.0e-8,maxits = 400, LOUD=1 )
print("k =",k)

plt.plot(x,phi_sol[:,0],'o',label="Group 1")
plt.plot(x,phi_sol[:,1],'o',label="Group 2")
plt.plot(x,phi_sol[:,-2],'o',label="Group 11")
plt.plot(x,phi_sol[:,-1],'o',label="Group 12")
plt.legend()
plt.show()
plt.plot(x,phi_thermal,label="Thermal")
plt.plot(x,phi_epithermal,label="Epithermal")
plt.plot(x,phi_fast,label="Fast")
plt.legend()
plt.show()
plt.semilogx(group_centers, phi_sol[I//2,:])
plt.show()


q = np.ones((I,G))*0
psi0 = np.zeros((I,N,G)) + 1e-12
group_mids = group_centers
dt_group = np.argmin(np.abs(group_mids-.025e-6)) #14.1))
print("14.1 MeV is in group", dt_group)
psi0[0,MU>0,dt_group] = 1
psi0[-1,MU<0,dt_group] = 1
numsteps = 100
dt = 5.0e-1
x,phi,psi = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,numsteps,dt,group_edges, tolerance = 1.0e-8,maxits = 200, LOUD=1 )
plt.plot(x,phi[:,0,-1])
plt.plot(x,phi[:,G-1,-1])
plt.show()

eigs,vsN,u = compute_alpha(psi,2,numsteps,I,G,N,dt)
upsi = np.reshape(u[:,0],(I,N,G))
uphi = np.zeros((I,G))
for g in range(G):
    for n in range(N): 
        uphi[:,g] += W[n] * upsi[:,n,g]
totneut = np.zeros(I)
for i in range(I):
    totneut[i] = np.sum(uphi[i,:]*inv_speed)

eigs,vsN,u = compute_alpha(psi,2,numsteps,I,G,N,dt)
u1psi = np.reshape(u[:,1],(I,N,G))
u1phi = np.zeros((I,G))
for g in range(G):
    for n in range(N): 
        u1phi[:,g] += W[n] * u1psi[:,n,g]
totneut1 = np.zeros(I)
for i in range(I):
    totneut1[i] = np.sum(u1phi[i,:]*inv_speed)
print("alphas = ", eigs)
x,alpha,phi_mode = multigroup_alpha(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,inv_speed,np.max(np.real(eigs)), tolerance = 1.0e-8,maxits = 100, LOUD=2 )