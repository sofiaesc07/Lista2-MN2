#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:19:19 2023

@author: GuillermoFurlan
"""
import numpy as np

def lu(A):
    
    
    n = len(A)  
    
    U = A.copy().astype(float)
    L = np.zeros((n, n))
    P = np.eye(n)

    for k in range(n - 1):
        
        i_max = k + np.argmax(np.abs(U[k:, k]))
        
        U[[k, i_max]] = U[[i_max, k]]
        L[[k, i_max]] = L[[i_max, k]]
        P[[k, i_max]] = P[[i_max, k]]

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    
    np.fill_diagonal(L, 1)

    return P, L, U

def cholesky(A):
    L = np.zeros_like(A)
    n = len(A)

    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            
            if (i == k): 
                L[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L

def QR(A):
    Q=[]
    R=[]
    V=[]
    for i in A:
        q=[0]*len(i)
        r=[0]*len(i)
        Q.append(q)
        R.append(r)
        v=[]
        for j in i:
            v.append(j)
        V.append(v)
    for i in range(len(A[0])):
        m=0
        for j in A:
            m=m+j[i]**2
        R[i][i]=m**(1/2)
        for j in range(len(A)):
            Q[j][i]=V[j][i]/R[i][i]
        for j in range(i+1,len(A[0])):
            pp=0
            for k in range(len(A)):
                pp=pp+Q[k][i]*V[k][j]
            R[i][j]=pp
            for k in range(len(A)):
                V[k][j]=V[k][j]-R[i][j]*Q[k][i]
    return np.array(Q),np.array(R)

def Positiva(A):
    """Verifica si una matriz es definida positiva"""
    return np.all(np.linalg.eigvals(A) > 0)

A=np.array([[1,0.5,1/3],[0.5,1/3,0.25],[1/3,0.25,0.2]])
B=np.array([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[-1,2,3,-1]])
C=np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,0]])

print("----------------------------------")
print("descomposicion LU")
print("A")
P,L,U=lu(A)
print("U",U)
print("L",L)
print("P",P)
print("")
print("")
print("B")
P,L,U=lu(A)
print("U",U)
print("L",L)
print("P",P)
print("")
print("")
print("c no tiene descomposicion LU al tener un 0 en su diagonal")
print("----------------------------------")
print("descompoicion de cholesky")
print("")
print("")
print("A")
L=cholesky(A)
print("L+",L)
print("")
print("B no es posotiva definida por lo que no tiene descomposicion de cholesky")

print("")
print("C no es posotiva definida por lo que no tiene descomposicion de cholesky")
print("-------------------------------------")
print("")
print("")
print("Descomoposicion QR")
Q,R=QR(A)
print("A")
print("Q=",Q)
print("R=",R)
print("")
print("")
Q,R=QR(B)
print("B")
print("Q=",Q)
print("R=",R)
print("")
print("")
Q,R=QR(C)
print("C")
print("Q=",Q)
print("R=",R)

