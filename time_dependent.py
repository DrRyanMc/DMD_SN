#!/usr/bin/python3

from ProcessData import NuclearData
import matplotlib.pyplot as plt
from multigroup_sn import *
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

#Compute atom densities
pct_c = 1e-6#0.01
atm_perc  = [1-pct_c,pct_c]
rho = [19.86, 2.1]
Na  = 0.60221  #in atoms-cm^2/(mol-b)
M = [239,12.01]
atm_dens = [rho[i]*Na/M[i]*atm_perc[i] for i in range(len(rho))]

#Process data
#    nd = NuclearData(["../crossx/Pu239_618gp_Pu239.cx","../crossx/h-fg_618gp_h-1.cx"],atm_dens=atm_dens)
nd = NuclearData(["./crossx/Pu239_12gp_Pu239.cx","./crossx/cnat_12gp_cnat.cx"],atm_dens=atm_dens)
metal_data = NuclearData(["./crossx/Pu239_12gp_Pu239.cx","./crossx/cnat_12gp_cnat.cx"],atm_dens=atm_dens)
metal_data.scat_mat[0] = np.transpose(metal_data.scat_mat[0])


pct_c = 1-1e-6
atm_perc  = [1-pct_c,pct_c]
c_dens = [rho[i]*Na/M[i]*atm_perc[i] for i in range(len(rho))]
print(c_dens)
refl = NuclearData(["./crossx/Pu239_12gp_Pu239.cx","./crossx/cnat_12gp_cnat.cx"],atm_dens=c_dens)
refl.scat_mat[0] = np.transpose(refl.scat_mat[0])

print(1/refl.inv_speed, refl.scat_mat[0])
plt.spy(refl.scat_mat[0])
plt.show()
I = 50
L = 10.8925 #11 super
Lx = L
hx = L/I
ref_thick = 8 #reflector thickness
#k = 0.995432853326 with ref_thick = 8 and L = 11 and I = 100
G = 12 #70
q = np.ones((I,G))
Xs = np.linspace(-hx/2,L-hx/2,I)
sigma_t = np.zeros((I,G))
nusigma_f = np.zeros((I,G))
chi = np.zeros((I,G))
sigma_s = np.zeros((I,G,G))
sigma_t[:,0:G] = metal_data.sig_t
sigma_s[:,0:G,0:G] = metal_data.scat_mat[0].transpose()
chi[:,0:G] = metal_data.chi
nusigma_f[:,0:G] = metal_data.nu_sig_f
inv_speed = metal_data.inv_speed
for i in range(I):
    if np.abs(Xs[i] - Lx/2) <= ref_thick/2:
        sigma_t[i,0:G] = refl.sig_t
        sigma_s[i,0:G,0:G] = refl.scat_mat[0].transpose()
        chi[i,0:G] = refl.chi
        nusigma_f[i,0:G] = refl.nu_sig_f

N = 4
MU, W = np.polynomial.legendre.leggauss(N)
BCs = np.zeros((N,G)) #-1
BCs[(N//2):N,:] = 0.0
#print(sigma_s)
"""
x, k, phi_sol, phi_thermal, phi_epithermal,phi_fast = multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs, metal_data,
                          tolerance = 1.0e-5,maxits = 100, LOUD=1 )
print("k =",k)

plt.plot(x,phi_thermal,'o',label="thermal")
plt.plot(x,phi_epithermal,'o',label="epithermal")
plt.plot(x,phi_fast,'o',label="fast")
plt.legend()
plt.show()

plt.semilogx(metal_data.groupCenters(), phi_sol[I//2,:])
plt.show()
assert 0
"""

q = np.ones((I,G))*0
psi0 = np.zeros((I,N,G)) + 1e-12
group_mids = metal_data.group_edges[0:G]+0.5*metal_data.group_de
dt_group = np.argmin(np.abs(group_mids-.025e-6)) #14.1))
print("14.1 MeV is in group", dt_group)
psi0[0,MU>0,dt_group] = 1
psi0[-1,MU<0,dt_group] = 1
#psi0[3*I//8:5*I//8,:,dt_group] = 1
numsteps = 20
dt = 5.0e-1
"""
x,phi,psi = multigroup_td(I,hx,G,sigma_t,(sigma_s),nusigma_f,chi,inv_speed,
                            N,BCs,psi0,q,numsteps,dt,metal_data, tolerance = 1.0e-8,maxits = 200, LOUD=1 )
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
    totneut[i] = np.sum(uphi[i,:]*metal_data.inv_speed)

eigs,vsN,u = compute_alpha(psi,2,numsteps,I,G,N,dt)
u1psi = np.reshape(u[:,1],(I,N,G))
u1phi = np.zeros((I,G))
for g in range(G):
    for n in range(N): 
        u1phi[:,g] += W[n] * u1psi[:,n,g]
totneut1 = np.zeros(I)
for i in range(I):
    totneut1[i] = np.sum(u1phi[i,:]*metal_data.inv_speed)
print("alphas = ", eigs)
"""
#-0.07924626
x, k, phi_old, alpha = multigroup_alpha(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,inv_speed,np.max(np.real(eigs)),-1, metal_data, tolerance = 1.0e-5,maxits = 100, LOUD=2 )