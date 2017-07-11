import sympy as sp

x,y,a,b,g,R = sp.symbols('x y a b g R')

Y = sp.Matrix(2,1,[x,g*y])
S = sp.Matrix(2,1,[a,b])
r = 1/(Y.dot(S))
I = sp.eye(2)
test = sp.Matrix(2,2,[2,0,2,0])
test2= sp.Matrix(2,2,[0.5,0,0.5,0])
#print test*test2

H = (I - R*S*Y.T)*(I-R*Y*S.T) + R*S*S.T
print (I - R*S*Y.T)*(I-R*Y*S.T)
#print (I-R*Y*S.T)
"""
print Y,S,r
print
#H=sp.simplify(H)
print 'Inverse Hessian'
print H[0,0]
print H[0,1]
print H[1,0]
print H[1,1]
"""
"""
H2 = (I - r*Y*S.T)*(I-r*S*Y.T) + r*Y*Y.T

print
print 'Hessian'
H2=sp.simplify(H2)
print H2[0,0]
print H2[0,1]
print H2[1,0]
print H2[1,1]

print sp.simplify(H*H2)

"""





