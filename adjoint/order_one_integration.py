import numpy as np


def exp_int(u,t):
    
    dt = t[1]-t[0]
    return dt*sum(u[:-1]) 

def imp_int(u,t):
    dt = t[1]-t[0]
    return dt*sum(u[1:])
