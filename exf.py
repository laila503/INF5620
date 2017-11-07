from fenics import *
import numpy as np
from exd import solve_numerical
import matplotlib.pyplot as plt

if __name__ == '__main__':
	def alpha(u):
		return 1.0 + u**2

	rho = 1.0
	f = Expression('-rho*pow(x[0],3)/3 + rho*pow(x[0],2)/2 + 8*pow(t,3)*pow(x[0],7)/9 - 28*pow(t,3)*pow(x[0],6)/9 + \
		 7*pow(t,3)*pow(x[0],5)/2 - 5*pow(t,3)*pow(x[0],4)/4 + 2*t*x[0] - t',t=0,rho=rho,degree=1)
	T = .3
	dt = 0.01

	#Define actual solution
	u_0 = Expression('0', degree=1)
	u_e = Expression('t*pow(x[0],2)*(1./2 - x[0]/3.)',t=0.0,degree=1)

	u,mesh,error = solve_numerical(T,dt,rho,u_0,u_e,f,alpha,dim="1D",P=1,meshsize=50)

	#For plotting exact solution
	x = np.linspace(0,1,int(1/dt))
	exact = T*x**2*(0.5-x/3.)

	plot(u)
	plot(mesh)
	interactive()
	plt.plot(x,exact)
	plt.title("Analytical solution for T = %g" %T)
	plt.xlabel("x")
	plt.ylabel("u(x)")
	plt.show()