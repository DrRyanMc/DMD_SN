#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:06:37 2018

@author: ryanmcclarren
"""

import numpy as np
import math
import matplotlib.pyplot as plt



def sweep1D(I,hx,q,sigma_t,mu,boundary):
    """Compute a transport sweep for a given
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        boundary:        value of angular flux on the boundary
    Outputs:
        psi:             value of angular flux in each zone
    """
    assert(np.abs(mu) > 1e-10)
    psi = np.zeros(I)
    ihx = 1/hx
    if (mu > 0): 
        psi_left = boundary
        for i in range(I):
            psi_right = (q[i] + (mu*ihx-0.5*sigma_t[i])*psi_left)/(0.5*sigma_t[i] + mu*ihx)
            psi[i] = 0.5*(psi_right + psi_left)
            psi_left = psi_right
    else:
        psi_right = boundary
        for i in reversed(range(I)):
            psi_left = (q[i] + (-mu*ihx-0.5*sigma_t[i])*psi_right)/(0.5*sigma_t[i] - mu*ihx)
            psi[i] = 0.5*(psi_right + psi_left)
            psi_right = psi_left
    return psi

def source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=False,psi_s =  0 ):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi:             value of scalar flux in each zone
    """
    phi = np.zeros(I)+1e-12
    phi_old = phi.copy()
    psi_mid = np.zeros((I,N))
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    oppmap = np.ndarray((N), dtype=int)
    for ord in range(N):
        oppmap[ord] = np.argmin(-MU[ord] - MU)
    iteration = 1
    psi_mid = np.zeros((I,N))
    iteration = 1
    while not(converged):
        phi = np.zeros(I)
        BCcopy = BCs.copy()
        #sweep over each direction
        for n in range(N):
            if BCs[n] < 0:
                if n < N//2:
                    BCcopy[n] = psi_mid[0, oppmap[n]]
                else:
                    BCcopy[n] = psi_mid[-1,oppmap[n]]
            tmp_psi = sweep1D(I,hx,q + phi_old*sigma_s*0.5 + psi_s[:,n],sigma_t,MU[n],BCcopy[n])
            psi_mid[:,n] = tmp_psi
            phi += tmp_psi*W[n]
        #check convergence
        assert not(math.isnan(phi[0]))
        change = np.linalg.norm(phi-phi_old)/np.linalg.norm(phi)
        converged = (change < tolerance) or (iteration > maxits)
        if (LOUD>0) or (converged and LOUD<-3):
            print("Iteration",iteration,": Relative Change =",change)
        if (iteration > maxits):
            print("Warning: Source Iteration did not converge")
        iteration += 1
        phi_old = phi.copy()
    x = np.linspace(hx/2,I*hx-hx/2,I)
    return x, phi, psi_mid

def source_iteration_td(I,hx,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=False,psi_s =  0 ):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi:             value of scalar flux in each zone
    """
    phi = np.zeros(I)
    phi_old = phi.copy()
    psi_mid = np.zeros((I,N))
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    oppmap = np.ndarray((N), dtype=int)
    for ord in range(N):
        oppmap[ord] = np.argmin(-MU[ord] - MU)
    iteration = 1
    psi_mid = np.zeros((I,N))
    iteration = 1
    while not(converged):
        phi = np.zeros(I)
        BCcopy = BCs.copy()
        #sweep over each direction
        for n in range(N):
            if BCs[n] < 0:
                if n < N//2:
                    BCcopy[n] = psi_mid[0, oppmap[n]]
                else:
                    BCcopy[n] = psi_mid[-1,oppmap[n]]
            tmp_psi = sweep1D(I,hx,q[:,n] + phi_old*sigma_s*0.5 + psi_s[:,n],sigma_t,MU[n],BCcopy[n])
            psi_mid[:,n] = tmp_psi
            phi += tmp_psi*W[n]
        #check convergence
        change = np.linalg.norm(phi-phi_old)/np.linalg.norm(phi)
        converged = (change < tolerance) or (iteration > maxits)
        if (LOUD>0) or (converged and LOUD<-3):
            print("Iteration",iteration,": Relative Change =",change)
        if (iteration > maxits):
            print("Warning: Source Iteration did not converge")
        iteration += 1
        phi_old = phi.copy()
    x = np.linspace(hx/2,I*hx-hx/2,I)
    return x, phi, psi_mid

def multigroup_ss(I,hx,G,q,sigma_t,sigma_s,nusigma_f,chi,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=False ):
    """Solve multigroup SS problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        G:               number of groups
        q:               source array
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    phi = np.zeros((I,G))
    phi_old = phi.copy()
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    oppmap = np.ndarray((N), dtype=int)
    for ord in range(N):
        oppmap[ord] = np.argmin(-MU[ord] - MU)
    iteration = 1
    psi_mid = np.zeros((I,N))
    psi_full = np.zeros((I,N,G))
    while not(converged):
        #phi = np.zeros((I,G))
        #solve each group
        if (LOUD > 0):
            print("Group Iteration",iteration)
            print("====================")
        for g in range(G):
            #compute scattering source
            Q = q[:,g].copy()
            BCcopy = BCs.copy()
            for gprime in range(G):
                Q += 0.5*(phi[:,gprime]*sigma_s[:,gprime,g]*(gprime != g) + chi[:,g] * phi[:,gprime] * nusigma_f[:,gprime])
            if (LOUD > 0):
                print("Group",g)
            x,phi[:,g],psi_mid = source_iteration(I,hx,Q,sigma_t[:,g],sigma_s[:,g,g],N,BCcopy[:,g], 
                                        tolerance = tolerance*0.1,maxits = 1000, LOUD=LOUD-1,
                                        psi_s = 0*psi_mid)
            
            psi_full[:,:,g] = psi_mid.copy()
        #check convergence
        change = np.linalg.norm(np.reshape(phi-phi_old,(I*G,1)))/np.linalg.norm(np.reshape(phi,(I*G,1)))
        converged = (change < tolerance) or (iteration > maxits)
        if (iteration > maxits):
            print("Warning: Group Iterations did not converge")
        if (LOUD>0) or (converged and LOUD<0):
            print("====================")
            print("Outer (group) Iteration",iteration,": Relative Change =",change)
            print("====================")
        iteration += 1
        phi_old = phi.copy()
    return x, phi

def multigroup_ss_td(I,hx,G,q,sigma_t,sigma_s,nusigma_f,chi,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=False):
    """Solve multigroup SS problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        G:               number of groups
        q:               source array
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    phi = np.zeros((I,G))
    phi_old = phi.copy()
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    oppmap = np.ndarray((N), dtype=int)
    for ord in range(N):
        oppmap[ord] = np.argmin(-MU[ord] - MU)
    iteration = 1
    psi_mid = np.zeros((I,N))
    Q = np.zeros((I,N))
    psi_full = np.zeros((I,N,G))
    while not(converged):
        #phi = np.zeros((I,G))
        #solve each group
        if (LOUD > 0):
            print("Group Iteration",iteration)
            print("====================")
        for g in range(G):
            #compute scattering source
            Q = q[:,:,g].copy()
            BCcopy = BCs.copy()
            for angle in range(N):
                for gprime in range(G):
                    Q[:,angle] += 0.5*(phi[:,gprime]*sigma_s[:,gprime,g]*(gprime != g) + chi[:,g] * phi[:,gprime] * nusigma_f[:,gprime])
            if (LOUD > 0):
                print("Group",g)
            x,phi[:,g],psi_mid = source_iteration_td(I,hx,Q,sigma_t[:,g],sigma_s[:,g,g],N,BCcopy[:,g], 
                                        tolerance = tolerance*0.1,maxits = 1000, LOUD=LOUD-1,
                                        psi_s = 0*psi_mid)
            psi_full[:,:,g] = psi_mid.copy()
        #check convergence
        change = np.linalg.norm(np.reshape(phi-phi_old,(I*G,1)))/np.linalg.norm(np.reshape(phi,(I*G,1)))
        converged = (change < tolerance) or (iteration > maxits)
        if (iteration > maxits):
            print("Warning: Group Iterations did not converge")
        if (LOUD>0) or (converged and LOUD<0):
            print("====================")
            print("Outer (group) Iteration",iteration,": Relative Change =",change)
            print("====================")
        iteration += 1
        phi_old = phi.copy()
    return x, phi, psi_full

def multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,group_edges=None, phi = np.zeros(1), tolerance = 1.0e-8,maxits = 100, LOUD=False ):
    """Solve k eigenvalue problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        G:               number of groups
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        nusigma_f:       array of nu times fission cross-sections format [i,g]
        chi:             energy distribution of fission neutrons
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    if (phi.size == 1) and (I != 1):
        phi = np.random.rand(I,G)
    phi = phi + np.flip(phi,0)
    phi_old = phi.copy()
    k = 1.0
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    iteration = 1
    while not(converged):
        #compute fission source
        Q = sigma_t*0.0
        for g in range(G):
            for gprime in range(G):
                Q[:,g] += 0.5*(chi[:,g] * nusigma_f[:,gprime] * phi_old[:,gprime])
        x,phi = multigroup_ss(I,hx,G,Q,sigma_t,sigma_s,0*sigma_t,0*sigma_t,N,BCs, tolerance = tolerance*0.001,maxits = maxits, LOUD=LOUD-1 )
        knew = np.linalg.norm(np.reshape(nusigma_f*phi,I*G))/np.linalg.norm(np.reshape(nusigma_f*phi_old,I*G))
        #check convergence
        solnorm = np.linalg.norm(np.reshape(phi_old,I*G))
        converged = ( ( (np.abs(knew-k) < tolerance)) or (iteration > maxits))
        if (LOUD>0) or (converged):
            print("*************************====================")
            print("Power Iteration",iteration,": k =",knew,"Relative Change =",np.abs(knew-k))
            print("*************************====================")
        iteration += 1
        k = knew
        phi_old = phi/k
    if (iteration > maxits):
        print("Warning: Power Iterations did not converge")
    
    #compute thermal flux, epithermal, and fast
    if (group_edges is not None):
        phi_thermal = np.zeros(I)
        phi_epithermal = np.zeros(I)
        phi_fast = np.zeros(I)
        for i in range(I):
            phi_thermal[i] = np.sum(phi_old[i,group_edges[0:G]<=(0.55e-6)])
            phi_fast[i] = np.sum(phi_old[i,group_edges[1:(G+1)]>=(1)])
            phi_epithermal[i] = np.sum(phi_old[i,:]) - phi_thermal[i] - phi_fast[i]
        return x, k, phi_old, phi_thermal, phi_epithermal,phi_fast
    return x, k, phi_old

def multigroup_alpha(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,inv_speed,min_alpha,max_alpha,group_edges=None,phi = np.zeros(1), tolerance = 1.0e-8,maxits = 100, LOUD=False, atol=1e-5 ):
    """Solve alpha eigenvalue problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        G:               number of groups
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        nusigma_f:       array of nu times fission cross-sections format [i,g]
        chi:             energy distribution of fission neutrons
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    if (phi.size == 1) and (I != 1):
        phi = np.random.rand(I,G)
    sigstar = sigma_t.copy()
    sigma_sstar = sigma_s.copy()
    print("min_alpha =", min_alpha )
    print("max_alpha =", max_alpha )
    #check signs
    alpha = min_alpha
    for i in range(I):
        if (alpha>=-10000):
            sigstar[i,:] = sigma_t[i,:] + alpha*inv_speed
        else:
            for g in range(G):
                sigma_sstar[i,g,g] = sigma_sstar[i,g,g] - alpha*inv_speed[g]
            
    x, k, phi_old, phi_thermal, phi_epithermal,phi_fast = multigroup_k(I,hx,G,sigstar,sigma_sstar,nusigma_f,chi,N,BCs,group_edges,phi, tolerance = tolerance*100,maxits = 100, LOUD=LOUD-1 )
    print("k at min alpha =", k)
    assert(k<1)
    
    
    alpha = max_alpha
    for i in range(I):
        if (alpha>=-10000):
            sigstar[i,:] = sigma_t[i,:] + alpha*inv_speed
        else:
            for g in range(G):
                sigma_sstar[i,g,g] = sigma_sstar[i,g,g] - alpha*inv_speed[g]
            
    x, k, phi_old, phi_thermal, phi_epithermal,phi_fast= multigroup_k(I,hx,G,sigstar,sigma_sstar,nusigma_f,chi,N,BCs,group_edges,phi, tolerance = tolerance*100,maxits = 100, LOUD=LOUD-1 )
    print("k at max alpha =", k)
    assert(k>1)
    converged = 0
    step = 0
    while not(converged):
        for i in range(I):
            if (alpha>=-10000):
                sigstar[i,:] = sigma_t[i,:] + alpha*inv_speed
            else:
                for g in range(G):
                    sigma_sstar[i,g,g] = sigma_sstar[i,g,g] - alpha*inv_speed[g]
                
        x, k, phi_old, phi_thermal, phi_epithermal,phi_fast = multigroup_k(I,hx,G,sigstar,sigma_sstar,nusigma_f,chi,N,BCs,group_edges,phi, tolerance = tolerance,maxits = 100, LOUD=LOUD-1 )
        if (k<1):
            min_alpha = alpha
        else:
            max_alpha = alpha
        alpha = 0.5*(max_alpha + min_alpha)
        step += 1
        converged = math.fabs(max_alpha - min_alpha) < atol
        print("Step", step,": alpha =", alpha, "k =", k, "[", min_alpha, ",", max_alpha, "]")
    return x, k, phi_old, alpha


def multigroup_td(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,inv_speed,N,BCs,psi0,qiso,numsteps,dt,group_edges=None, tolerance = 1.0e-8,maxits = 100, LOUD=False ):
    """Solve k eigenvalue problem
    Inputs:
        I:               number of zones 
        hx:              size of each zone
        G:               number of groups
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        nusigma_f:       array of nu times fission cross-sections format [i,g]
        chi:             energy distribution of fission neutrons
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    phi_out = np.zeros((I,G,numsteps))
    psi_out = np.zeros((I,N,G,numsteps))
    phi = np.zeros((I,G))
    psi = np.zeros((I,N,G))
    if (psi0.size == I*N*G):
        psi = psi0.copy()
    phi_old = phi.copy()
    MU, W = np.polynomial.legendre.leggauss(N)
    iteration = 1
    q = np.zeros((I,N,G))
    Q = np.zeros((I,N,G))
    for angle in range(N):
        q[:,angle,:] = qiso
    for step in range(numsteps):
        #compute fission source
        for angle in range(N):
            Q[:,angle,:] = q[:,angle,:] + psi[:,angle,:]*inv_speed/dt
        
        x,phi_out[:,:,step], psi = multigroup_ss_td(I,hx,G,Q,sigma_t+1/dt*inv_speed,sigma_s,nusigma_f,chi,N,BCs, tolerance = tolerance*0.01,maxits = 100, LOUD=LOUD-1 )
        psi_out[:,:,:,step] = psi.copy()
        if (LOUD>=1):
            print("**********************\nStep", step,"t =",(step+1)*dt,"\n**********************")
            #plt.plot(x,phi_out[:,:,step])
            #plt.show()
            
            if (group_edges is not None):
                #compute thermal flux, epithermal, and fast
                phi_thermal = np.zeros(I)
                phi_epithermal = np.zeros(I)
                phi_fast = np.zeros(I)
                for i in range(I):
                    phi_thermal[i] = np.sum(phi_out[i,group_edges[0:G]<=(0.55e-6),step])
                    phi_fast[i] = np.sum(phi_out[i,group_edges[1:(G+1)]>=(1),step])
                    phi_epithermal[i] = np.sum(phi_out[i,(group_edges[1:(G+1)]<=(1)) * (group_edges[0:G]>=(0.05e-6)),step])
                plt.semilogy(x,phi_thermal, label="thermal")
                plt.semilogy(x,phi_epithermal, label="epithermal")
                plt.semilogy(x,phi_fast,label="fast")
                plt.legend(loc="best")
                plt.show()
        
    return x, phi_out, psi_out