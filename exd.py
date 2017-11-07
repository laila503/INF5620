"""
Solving the nonlinear diffusion equation:

	rho*u_t = grad dot (aplha(u)*grad(u) + f(vec(x),t)
	u(vec(x),0) = I(x)
	partial(u)/partial(n) = 0 (Newman boundary condition)

"""

from fenics import *
import numpy as np

def solve_numerical(T,dt,rho,u_0,u_e,f,alpha,dim="2D",P=1,meshsize=32):

	#Verification values for inputdata:

	nT = int(float(T)/dt)
	time_array = np.linspace(0,T,nT)


	#Create mesh:
	if dim == "1D":
		mesh = UnitIntervalMesh(meshsize)
	elif dim == "2D":
		mesh = UnitSquareMesh(meshsize,meshsize)
	else:
		mesh = UnitCubeMesh(meshsize,meshsize,meshsize)


	V = FunctionSpace(mesh, 'P',P)


	u = TrialFunction(V)
	v = TestFunction(V)


	#define boundary condition
	def u0_boundary(x, on_boundary):
		return on_boundary

	bc = DirichletBC(V, u_0, u0_boundary)

	u_n = interpolate(u_0, V) #previous u

	#Define the variational problem
	a = inner(u,v)*dx + (dt/rho)*inner(alpha(u_n)*grad(u),grad(v))*dx
	L = inner(u_n,v)*dx + (dt/rho)*inner(f,v)*dx
	

	#Array for the error
	error = np.zeros(nT)

	#Compute solution
	A = assemble(a)
	
	u = Function(V)

	for i,t in enumerate(time_array):
		#b = assemble(L)
		solve(a==L, u)

		u_n.assign(u)

		u_e.t = t
		u_e_in_V =interpolate(u_e,V)
		e = u_e_in_V.vector().array()-u.vector().array()
		error[i] = np.sqrt(np.sum(e**2)/u.vector().array().size)

	return u,mesh,error

def test_numerical_with_constant():
	def alpha(u):
		return 1.0 
	dt = 0.01
	rho = 1.0
	f = Constant("0.0")
	T = 1

	#Define actual solution
	u_0 = Expression('4.0', degree=1)
	u_e = Constant("4.0")

	u,mesh,error = solve_numerical(T,dt,rho,u_0,u_e,f,alpha,dim="2D",P=1)


	#Plot solution
	u.rename('u','Solution')
	plot(u)
	plot(mesh)
	interactive()

	eps = 1e-9
	assert abs(np.sum(error) < eps), "Something went wrong!"
	

if __name__ == '__main__':

	test_numerical_with_constant()
	




