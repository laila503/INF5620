
from fenics import *
import numpy as np
from exd import solve_numerical
import matplotlib.pyplot as plt

if __name__ == '__main__':
	def alpha(u):
		return 1.0 

	rho = 1.0
	f = Constant("0.0")
	T = 0.05

	#Define actual solution
	u_0 = Expression('cos(pi*x[0])',pi=pi, degree=1)
	u_e = Expression('cos(pi*x[0])*exp(-t*pi*pi)',pi=pi,t=0.0,degree=1)

	meshsize = [m for m in range(30,71,10)]
	E_over_h = np.zeros(len(meshsize))
	hs = np.zeros_like(E_over_h)

	for i,m in enumerate(meshsize):
		dt = h = 1./float(m)**2
		u,mesh,error = solve_numerical(T,dt,rho,u_0,u_e,f,alpha,dim="2D",P=1,meshsize=m)
		E_over_h[i] = error[-1]/float(h)
		hs[i] = h

	
	plt.plot(hs,E_over_h)
	plt.title("E/h for varying h for T =%g." %T)
	plt.xlabel("h")
	plt.ylabel("E/h")
	plt.show()