#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import numba
from numba import jit

import copy

@jit(nopython=True)
def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return int(k/ncol), k % ncol

# @jit(nopython=True)


def FindMST(A):

    p = A.shape[0]
    MST = np.zeros((p, p))

    Xl = list()
    Vl = list(range(p))
    Xl.append(0)
    Vl.remove(0)

    i = 0
    while len(Vl) > 0:
        idx0, idx1 = find_min_idx(A[Xl][:, Vl])

        MST[Xl[idx0], Vl[idx1]] = 1

        Xl.append(Vl[idx1])
        Vl.remove(Vl[idx1])

        i += 1

    return MST + MST.T


def getB(A):
    p = A.shape[0]

    B = np.zeros([p, p-1])

    idx = 0
    for i in range(p):
        for j in range(i):
            if A[i, j] == 1:
                B[i, idx] = 1
                B[j, idx] = -1
                idx += 1
    return B


@jit(nopython=True)
def getA(B):
    A = - B@B.T
    np.fill_diagonal(A, 0)
    return A


@jit(nopython=True)
def getSubInverse(B2Inv, k,full_edge_idx):

    sel = full_edge_idx == k
    sel_not = full_edge_idx != k

    M11 = B2Inv[sel_not, ][:, sel_not]
    M12 = B2Inv[sel_not, ][:, sel]
    M22 = B2Inv[sel, ][:, sel]

    return M11 - M12@M12.T/M22[0, 0]


@jit(nopython=True)
def findRowColIdx(idx, m1, m2):
    r = int(idx / m2)
    return (r,     idx - r*m2)

@jit(nopython=True)
def logEdgeDensity(EdgeD, tau,n, mu_tau,alpha):
    
    return (-n* np.log(tau) - (alpha+ n) * np.log(1.0+ EdgeD/tau)).sum() - tau/mu_tau



@jit(nopython=True)
def updateB(B, alpha, tau, B2Inv,p,n, D):

    full_edge_idx = np.arange(p-1)
    full_node_idx = np.arange(p)

    for k in range(p-1):

        # quick way to compute P_s and update B2inv

        sel = full_edge_idx == k
        sel_not = full_edge_idx != k
        B_not_s = B[:, sel_not]

        B_not_s2_inv = getSubInverse(B2Inv, k, full_edge_idx)


#         B_not_s2_inv_B_not_s = B_not_s2_inv@B_not_s.T    # complexity O(p^3)
#         P_s = np.eye(p) - B_not_s @ B_not_s2_inv_B_not_s # complexity O(p^3)
#         beta_k = P_s@(B[:,sel])

        # complexity O(p^2)
        beta_k = B[:, sel] - B_not_s@(B_not_s2_inv@ (B_not_s.T@B[:, sel]))

#         beta_k = NulSpaceB(B,k)
#         beta_k = np.round(beta_k.flatten(),2)

        beta_k = beta_k.flatten()
        mid_point = (np.max(beta_k)+np.min(beta_k))/2

        subgraph0 = beta_k > mid_point
        subgraph1 = beta_k <= mid_point

#         beta_k = np.floor(beta_k.flatten()* 1E10)

#         unique_values = np.unique(beta_k) #list(set(beta_k))

#         if len(unique_values)!=2:
#             print(unique_values)
#             print("error")

#         subgraph0 = beta_k==unique_values[0]
#         subgraph1 = beta_k==unique_values[1]

        subgraph0_node_idx = full_node_idx[subgraph0]
        subgraph1_node_idx = full_node_idx[subgraph1]

        logp_choices = -(alpha+ n)*np.log(1.0+ D[subgraph0_node_idx,:][:, subgraph1_node_idx]/tau)

        m1, m2 = logp_choices.shape
        logp_gumbel = logp_choices + np.random.gumbel(0, 1, logp_choices.shape)
        idx = np.argmax(logp_gumbel)

        r_idx, c_idx = findRowColIdx(idx, m1, m2)

        B_k_vec = (B[:, k]).copy()

        new_idx0 = subgraph0_node_idx[r_idx]
        new_idx1 = subgraph1_node_idx[c_idx]

        B[:, k] = 0
        B[new_idx0, k] = 1
        B[new_idx1, k] = - 1

        if np.sum(np.abs(B[:, k] - B_k_vec)) > 0:

            # update B2Inv
            B_s_star = B[:, sel]
#             M22_star = 1/(B_s_star.T@P_s@B_s_star) # complexity: O(p^3)
            b = B_not_s.T@B_s_star  # complexity: O(p^2)
            B_invB_b = B_not_s2_inv@b  # complexity: O(p^2)
            M22_star = 1./(2.0 - b.T@(B_invB_b))


#             M12_star= - B_not_s2_inv_B_not_s@B_s_star * M22_star
            M12_star = - B_invB_b * M22_star

            M11_star = B_not_s2_inv+M12_star@M12_star.T/M22_star

            temp = B2Inv[sel_not]
            temp[:, sel_not] = M11_star
            temp[:, sel] = M12_star
            B2Inv[sel_not] = temp

            temp = B2Inv[sel]
            temp[:, sel_not] = M12_star.T
            temp[:, sel] = M22_star
            B2Inv[sel] = temp

    return [B, B2Inv]

def updateTau(EdgeD, tau,n, mu_tau,alpha, step_size):
    
    cur_tau = tau
    prop_tau = np.abs( tau + np.random.uniform(low= -1.0, high=1.0, size=1)* step_size)
    
    cur_l = logEdgeDensity(EdgeD, cur_tau,n, mu_tau,alpha)
    prop_l = logEdgeDensity(EdgeD, prop_tau,n, mu_tau,alpha)
    
    log_u = np.log(np.random.uniform(0,1,1))
    
#     print(prop_l-cur_l)
    
    if( log_u<  (prop_l-cur_l)):
        accept =1
        return ([prop_tau, accept])
    else:
        accept =0
        return ([cur_tau, accept])



class SpanningTree:
    def __init__(self, Y):
        super(SpanningTree, self).__init__()

        self.Y = Y
        self.p = p = Y.shape[0]
        self.n = n = Y.shape[1]
        D = np.zeros([p, p])
        for i in range(p):
            for j in range(i):
                D[i, j] = np.sqrt(np.sum((Y[i]-Y[j])**2))
                D[j, i] = D[i, j]

        self.D = D
        self.mst0 = FindMST(D)
        B = getB(self.mst0)

        B2Inv = np.linalg.inv(B.T@B)


        tau = np.min(D[D>0]) / self.n
        
        self.mu_tau = tau
        
        self.alpha = 5.0
        
        EdgeD = -np.diag(B.T@D@B)/2
        

        self.params = [B, B2Inv, EdgeD, tau]

    def computeMarginalProb(self):
        
        [B, B2Inv, EdgeD, tau] = self.params
        p = self.p
        n = self.n
        
        A = self.alpha/2/tau * (1+ self.D/tau)**(- (self.alpha + n))
        
#         np.fill_diagonal(A,0.0)
        
        L = np.diag(A.sum(1))-A
        L_inv = np.linalg.inv(L+ 1.0/p + np.eye(p)*1E-12)

        L_inv_diag = np.diag(L_inv).reshape([p,1])

        prob = -(2* L_inv - L_inv_diag - L_inv_diag.T)*A
        
        return prob
        
        
    def runMCMC(self, steps= 100, step_size = 1E-3):

        [B, B2Inv, EdgeD, tau] = self.params
        p = self.p
        n = self.n
        D = self.D
        
        mu_tau = self.mu_tau

        full_edge_idx = np.arange(p-1)
        full_node_idx = np.arange(p)

        trace = list()
        
        accept_tau_count = 0

        for s in range(steps):

            B, B2Inv = updateB(B, self.alpha, tau, B2Inv,p,n, D)
            EdgeD = -np.diag(B.T@D@B)/2
            # A = getA(B)
            
            
            tau, accept_tau = updateTau(EdgeD, tau,n, mu_tau, self.alpha, step_size)
            accept_tau_count += accept_tau
        

            self.params = [B, B2Inv, EdgeD, tau]

            trace.append(copy.deepcopy(self.params))

            if (s+1) % 100==0:
                print(s)
                

        print( float(accept_tau_count)/steps)

        return trace

    def extractGraph(self, trace):

        trace_A= [-x[0]@np.diag(1-x[6])@x[0].T for x in trace]

        for i in range(len(trace)):
            np.fill_diagonal(trace_A[i],0)

        return trace_A


