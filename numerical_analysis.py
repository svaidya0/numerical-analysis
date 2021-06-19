import numpy as np
from numpy.core.numeric import False_
from numpy.linalg import solve, norm
from matplotlib import pyplot as plt

# One step Forward Euler Method
def forwardEuler(f,Df, t_n,y_n, h):
    y = y_n + h*f(t_n,y_n)
    return y

# One step Newton-Raphson Method
def newton (F, DF, x0, eps,K):
    x = x0
    for k in range(1,K+1):
        d = solve(DF(x), F(x))
        x_k1 = x-d
        if norm(F(x_k1)) < eps:
            return x_k1,k
        else:
            x = x_k1
    return x, k

# One step Huen Method
def huen(f,Df, t_n,y_n, h):
    f_tn = f(t_n, y_n)
    y = y_n + 0.5*h*f_tn + 0.5*h*f(t_n + h, y_n + h*f_tn)
    return y

# One step Backward Euler Method
def backwardEuler(f,Df, t0,y0, h):
    F = lambda d: d - f(t0+h, y0 + h*d)
    m = len(F(y0))
    dfy = lambda d: np.identity(m) - (h * Df(t0+h, y0 + h*d))
    x,_ = newton(lambda d: F(d), lambda d: dfy(d), f(t0, y0), 1e-12, 10000)
    y = y0 + h*x
    return y

# One step Crank-Nicholson Method
def crank_nichol(f,Df, t_n,y_n, h):
    f_tn = f(t_n, y_n)
    F = lambda d: d - y_n - 0.5*h*(f_tn + f(t_n + h, d))
    m = len(F(y_n))
    dfy = lambda d: np.eye(m) - 0.5*h*(Df(t_n + h, d))
    x,_ = newton(F, dfy, y_n+h, 1e-12, 10000)
    return x

# One step second order Taylor Method
def taylor(f, Df, dtdf, t0, y0, h):
    y = y0 + h*f(t0,y0) + 0.5*(h**2)*(dtdf(t0, y0) + np.dot(Df(t0, y0), f(t0, y0)))
    return y

# Function computes the approximation y_n≈Y(t_n) with n=0,…,N to the solution Y of an initial value problem
# y′(t)=f(t,y(t)) ,y(0)=y0
# with h=T/N and y_(n+1) = phi(tn,yn;h,f) where phi is one of the above methods
def evolve(phi, f,Df, t0,y0, T,N):
    h = T/N
    y = np.zeros([N+1, len(y0)])
    y[0] = y0
    t_n = t0
    for i in range(1, N+1):
        t_n = t_n + h 
        y[i] = phi(f, Df, t_n, y[i-1], h)   
    return y

# Function to compute the experimental order of convergence (EOC) for a given sequence of step sizes and errors given in the form of a mx2 matrix ((h1,e1),(h2,e2),...,(hm,em))
# Returns a vector of EOCs values given by eoc_i = log( e_(i+1)/e_i ) / log( h_(i+1)/h_i )
def computeEocs( herr ):
    r = len(herr)
    eocs = np.zeros(r-1)
    for i in range(r-1):
        eocs[i] = np.log(herr[i+1][1] / herr[i][1]) / np.log(herr[i+1][0] / herr[i][0])
    return eocs

# One step Diagonally Implicit Runge-Kutta Method where alpha, beta, gamma represent the Butcher Tableau
# e.g. alpha, beta gamma for Huen's Method are:
    """ alpha = np.array([0.0, 1.0])
    beta = np.array([[0.0, 0.0], [1.0, 0.0]])
    gamma = np.array([0.5, 0.5]) """
def dirk(f,Df, t0,y0, h, alpha,beta,gamma):
    f0 = f(t0, y0)
    s = len(alpha)
    m = len(y0)
    
    kap = np.zeros((s, m))
    phi = 0
    F = lambda kappa: kappa - f(t0+h*alpha[i], y0 + h*beta[i][i]*kappa + h*np.dot(beta[i][0:i],kap[0:i]))
    DF = lambda kappa: np.eye(m) - h*Df(t0+h*alpha[i], y0 + h*beta[i][i]*kappa + h*np.dot(beta[i][0:i],kap[0:i]))
    eps = 1e-16
    for i in range(s):
        kap[i] = f0
        k  = 0
        K = 100000
        Fx = F(kap[i])
        while Fx.dot(Fx) > eps*eps and k<K:
            kap[i] -= solve(DF(kap[i]), Fx)  # don't construct a new vector in each step - they could be large
            Fx = F(kap[i])
            k += 1
        phi += gamma[i] * kap[i]
    y = y0 + h*phi
    return y


#####
# Example - take the function:
# y′(t) = (c−y(t))^2, y(0) = 1, c >0 
# which has the exact solution Y(t) = (1 + tc(c−1)) / (1 + t(c−1))
# We compute the errors at the final time T and the EOC for the sequence of time steps given by the Crank-Nicholson Method with
# h_i = T/N_i with N_i = 20*2^i for i = 0,1,...,10 and c = 1.5, T = 10

c = 1.5
T = 10
N_0 = 20
Y = (1 + (T*c*(c-1)) ) / (1 + (T*(c-1)) )
herr = np.zeros((11, 2))
err = np.zeros(11)

def implement_CN(c, T, N_0, Y, phi, f, Df):
    y0 = np.array([1.0])
    herr = np.zeros((11, 2))
    err = np.zeros(11)

    for i in range(11):
        N = N_0 * (2**i)
        y = evolve(phi, f, Df, 0, y0, T, N)
        err[i] = abs(y[N] - Y)
        herr[i][0] = T/N
        herr[i][1] = err[i]

    eocs = computeEocs(herr)
    h = np.zeros(11)

    # Prints i, N, h, error, and EOC
    print('{:<3}{:>6}{:^20}{:^30}{:^30}'.format("i","N", "h", "|Y(T) - Y_N|", "EOC"))
    for i in range(11):
        N = N_0 * (2**i)
        h[i] = T/N
        if i == 0:
            print('{:<3}{:>6}{:^20}{:^30}{:^30}'.format(i, N, h[i], err[i], "N/A"))
            continue
        print('{:<3}{:>6}{:^20}{:^30}{:^30}'.format(i, N, h[i], err[i], eocs[i-1]))
    print("\n")

    if phi == crank_nichol:
        plt.plot(h, err, label = "Crank-Nicolson", marker = 'x')
    else:
        plt.plot(h, err, label = "DIRK with Crank-Nicolson", marker = 'o')

print("\nCrank-Nicholson Mehtod\n")
f = lambda t,y: np.array([(c-y)**2])
Df = lambda t,y: np.array([-2*(c-y)])
y0 = np.array([1.0])
implement_CN(c, T, N_0, Y, crank_nichol, f, Df)

plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Step Size", fontsize = 20)
plt.ylabel("Error", fontsize = 20)
plt.show(block=False)             # Prevents script from pausing
plt.figure()

# For Crank_Nicholson using DIRK:
print("Crank-Nicholson Mehtod under DIRK\n")
alpha = np.array([0.0, 1.0])
beta = np.array([[0.0, 0.0], [0.5, 0.5]])
gamma = np.array([0.5, 0.5])
f = lambda t,y: (c-y)**2
Df = lambda t,y: -2*(c-y)
stepper = lambda f,Df,t0,y0,h: dirk(f,Df,t0,y0,h,alpha,beta,gamma)
implement_CN(c, T, N_0, Y, stepper, f, Df)

plt.yscale('log')
plt.xscale('log')
plt.xlabel("Step Size", fontsize = 20)
plt.ylabel("Error", fontsize = 20)
plt.legend()
plt.show(block=False)             # Prevents script from pausing
plt.show()