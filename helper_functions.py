''' Code written by C. Totzeck, Dec 2023

    Run the code to obtain the data and results of the first test case presented in the article
    'Data-driven adjoint-based calibration of port-Hamiltonian systems in time domain'
    by Michael Günther, Birgit Jacob and Claudia Totzeck (IMACM, University of Wuppertal)

'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as spl
import copy
import scipy.signal as scs

matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)

# activate tex expressions in lables of figures
plt.rcParams['text.usetex'] = True

# fix random seed to make results reproducable
np.random.seed(10)

############################################
#### general functions ####
############################################
def update_if_successful(JA,RA,QA,BA,x0A,xA,yA,cn,cA,J, R, Q, B, x0, xn, yn, cost_last,fail):

    if fail:
        print('Armijo failed')
        return J, R, Q, B, x0, xn, yn, cost_last, cost_last
    else:

        cost_last = copy.deepcopy(cn)

        return JA, RA, QA, BA, x0A, xA, yA, cost_last, cA

def Q_transp(Q1,Q2,XI):
    E = spl.sqrtm(spl.solve_sylvester(Q2, np.zeros(Q2.shape), Q1));
    return E.dot(XI.dot(E.transpose()))

def J_transp(J1,J2,XI):
    return XI

def B_transp(B1,B2,XI):
    return XI

def R_transp(R1,R2,XI):
    XI = XI - (XI[:].dot(R2[:]))*R2
    YtY = (R2.transpose()).dot(R2)
    AS = (R2.transpose()).dot(XI) - (XI.transpose()).dot(R2)
    Omega = spl.solve_sylvester(YtY, YtY, AS)
    XIproj = XI - R2.dot(Omega)
    return XIproj

def x_transp(x1,x2,XI):
    return XI

def transport(J,JA,dir_J, R, RA, dir_R, Q, QA, dir_Q, B, BA, dir_B, X, XA, dir_X):
    # transport to new tangent space for CG directions
    XI_J = J_transp(J, JA, dir_J)
    XI_R = R_transp(R, RA, dir_R)
    XI_Q = Q_transp(Q, QA, dir_Q)
    XI_B = B_transp(B, BA, dir_B)
    XI_x = x_transp(X, XA, dir_X)

    return XI_J, XI_R, XI_Q, XI_B, XI_x

def CG_direction(Grad_J,Grad_R,Grad_Q,Grad_B,Grad_x, XI_B, XI_Q, XI_J, XI_R, XI_x, n1, norm_old, i, N):

    if i == 0 or i%5 == 0:
        #print('gradient is used!')
        dir_J = -Grad_J
        dir_R = -Grad_R
        dir_Q = -Grad_Q
        dir_B = -Grad_B
        dir_x = -Grad_x

    else:
        n2 = compute_norm(XI_B, XI_Q, XI_J, XI_R, XI_x)
        s = min(1, n1 / n2)
        beta = n1 ** 2 / norm_old ** 2

        dir_J = -Grad_J - beta * s * XI_J
        dir_R = -Grad_R - beta * s * XI_R
        dir_Q = -Grad_Q - beta * s * XI_Q
        dir_B = -Grad_B - beta * s * XI_B
        dir_x = -Grad_x - beta * s * XI_x

    return dir_J, dir_R, dir_Q, dir_B, dir_x


def initialize_container_postprocessing():
    # containers for post-processing
    cost_vec = []
    norm_vec_Q = []
    norm_vec_J = []
    norm_vec_R = []
    norm_vec_B = []

    return cost_vec, norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B

def print_reference(BT,QT,JT,RT,xT):

    print('-----------------------------')
    print('----- reference values ------')
    print('BT',BT)
    print('QT',QT)
    print('JT',JT)
    print('RT',RT)
    print('xT',xT)
    print('-----------------------------')

def compute_norms(QT,Q,JT,J,RT,R,BT,B,norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B):
    norm_vec_Q.append(np.sqrt(np.sum(np.power(QT - Q, 2))))
    norm_vec_J.append(np.sqrt(np.sum(np.power(JT - J, 2))))
    norm_vec_R.append(np.sqrt(np.sum(np.power(RT - R, 2))))
    norm_vec_B.append(np.sqrt(np.sum(np.power(BT - B, 2))))

    return norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B


def generate_symm(N,fix_seed=0):

    if fix_seed>0:
        np.random.seed(fix_seed)

    A = -0.1 + 0.2*np.random.rand(N,N)

    tud = np.triu(A, k=0) # upper part of matrix including diagonal

    tud1 = np.triu(A, k=1) # upper part of matrix without diagonal

    S = tud + tud1.T # symmetric matrix

    return S

def test_symm(S):
    flag = (np.sum(S-S.T) == 0)
    # if flag:
    #      print('Symm OK!')
    # else:
    #      print('WARNING: Matrix not symmetric!')
    return flag

def generate_skew_symm(N,fix_seed=0):

    if fix_seed>0:
        np.random.seed(fix_seed)

    A = -0.1 + 0.1*np.random.rand(N,N)

    tud = np.triu(A, k=1) # upper part of matrix without diagonal / diagonal zero real-valued skrew symm

    tud1 = np.triu(A, k=1) # upper part of matrix without diagonal

    S = tud - tud1.T # symmetric matrix

    return S

def test_skew(S):
    flag = (np.sum(S + S.T) == 0)
    # if np.sum(S + S.T) == 0:
    #     print('Skrew OK!')
    # else:
    #     print('WARNING: Matrix not skrew-symmetric!')
    return flag

def solve_forward(x0,J,R,Q,B,u,steps,N,dt):

    xn = np.zeros([steps + 1, N])
    xn[0,:] = x0
    yn = np.zeros(u.shape)
    # initial values
    # start with zero
    yn[0, :] = np.dot(B.transpose(), np.dot(Q, xn[0, :]))

    t = 0
    i=1
    while i<steps+1:

        # Euler step
        xn[i,:] = xn[i-1,:] + np.dot(J-R,np.dot(Q,xn[i-1,:]))*dt + np.dot(B,u[i-1,:])*dt
        yn[i, :] = np.dot(B.transpose(), np.dot(Q, xn[i, :]))

        t += dt
        i += 1
    return xn, yn

def solve_backward(J,R,Q,B,y,ydata,sol_x,T,dt,steps):

    pn = np.zeros(sol_x.shape)

    t = T+dt
    i = steps-1
    while t > 0:
        # Euler step
        pn[i,:] = pn[i+1,:] + (np.dot(np.dot(Q.T,J.T-R.T),pn[i+1,:]) + np.dot(np.dot(Q.transpose(),B),y[i+1,:]- ydata[i+1,:]))*dt
        t -= dt
        i -= 1

    return pn

def make_symmetric(S):

    sym = 0.5*(S + S.T)

    return sym

def make_skew(S):

    skew = 0.5*(S - S.T)

    return skew

def compute_gradient_B(x,y,p,Q,B,u,ydata,steps,dt):

    G2 = np.zeros(B.shape)
    for i in range(steps):
        G2 = G2 + dt * np.outer(p[i, :], u[i, :]) + dt*np.outer(np.dot(Q,x[i,:]),y[i,:]-ydata[i,:])
    return G2

def compute_gradient_Q(x,y,p,Q,J,R,B,u,ydata,steps,dt,lam=0.0):
    IR = spl.inv(Q)
    G1 = lam*IR.dot(spl.logm(Q))
    G2 = np.zeros(Q.shape)
    for i in range(steps):
        G2 = G2 + dt * np.outer(x[i, :], np.dot(J.transpose()-R.transpose(),p[i, :])) + dt * np.outer(x[i, :], np.dot(B,y[i, :]-ydata[i,:]))
    return G1+G2

def compute_gradient_J(x,p,J,Q,steps,dt):

    G2 = np.zeros(J.shape)
    for i in range(steps):
        G2 = G2 + dt * np.outer(p[i, :],np.dot(Q,x[i, :]))
    return G2

def compute_gradient_R(x,p,R,Q,steps,dt):
    G2 = np.zeros(R.shape)
    for i in range(steps):
        G2 = G2 - dt * np.outer(p[i, :],np.dot(Q,x[i, :]))
    return G2

def compute_gradient_x(p,steps,dt):
    return p[0,:]

def compute_gradient(xn,yn,pn,Q,B,J,R,u,ydata,steps,dt,lam):

    # compute components of gradient
    Grad_B = compute_gradient_B(xn,yn,pn,Q,B,u,ydata,steps,dt)
    Grad_Q = compute_gradient_Q(xn,yn,pn,Q,J,R,B,u,ydata,steps,dt,lam)
    Grad_J = compute_gradient_J(xn,pn,J,Q,steps,dt)
    Grad_R = compute_gradient_R(xn,pn,R,Q,steps,dt)
    Grad_x = compute_gradient_x(pn,steps,dt)

    return Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x

def compute_retraction_x(x0,xi0):
    return x0 + xi0

def compute_retraction_B(R,XI):
    return R+XI

def compute_retraction_Q(Q,XI,alternative=True):
    helpG = spl.solve_sylvester(Q,np.zeros(Q.shape),XI)
    G = make_symmetric(Q + XI + 0.5*np.dot(XI,helpG) )
    return G

def compute_retraction_J(J,XI):
    return J+XI

def compute_retraction_R(R,XI):
    return R+XI


def evaluate_cost(y,ydata,lam,Q,dt):
    return 0.5*np.sum(np.power(y-ydata,2))*dt + lam*np.linalg.norm(spl.logm(Q),'fro')


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def compute_norm(B, Q, J, R, x):
    return np.linalg.norm(B,'fro') + np.linalg.norm(Q,'fro') + np.linalg.norm(J,'fro') + np.linalg.norm(R,'fro') + np.linalg.norm(x)



def check_matrix_properties(dir_Q, dir_R, dir_J):

    if ~test_symm(dir_Q):
       dir_Q = make_symmetric(dir_Q)

    if ~test_symm(dir_R):
       dir_R = make_symmetric(dir_R)

    if ~test_skew(dir_J):
       dir_J = make_skew(dir_J)

    return dir_Q, dir_R, dir_J


def compute_inner_B(B,D1,D2):

    ip = np.sum(np.multiply(D1,D2))

    return ip

def compute_inner_Q(Q,D1,D2):

    X1 = spl.solve_sylvester(Q,np.zeros(Q.shape),D1)
    X2 = spl.solve_sylvester(Q,np.zeros(Q.shape),D2)
    ip = np.trace(np.dot(X1,X2))

    return ip

def compute_inner_J(J,D1,D2):

    ip = np.sum(np.multiply(D1,D2))

    return ip

def compute_inner_R(R,D1,D2):

    ip = np.trace(np.dot(D1,D2))

    return ip

def compute_inner_x(x,D1,D2):

    ip = np.sum(np.multiply(D1,D2))

    return ip

############################################
#### functions for gradient check ####
############################################
def initialize_solver_parameters(steps,N,T):

    # choose if alternative retraction for Q is used
    alternative = False  # True /False

    # armijo parameter
    gamma = 0.0001
    sig = 1000.
    fail_flag = False
    eps = 0.00000001

    # ODE solution
    sol_x = np.zeros((steps + 1, N))

    # reference solution
    ref_x = np.zeros((steps + 1, N))

    # time vec
    t_vec = np.linspace(0, T, steps + 1)

    return alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec

def initialilze_problem_parameters(noiseFlag=False):
    # the determinisitic version runs without noise
    noise_flag = noiseFlag  # True/False

    # fix dimension of the matrices
    M = 2  # dimension of control
    N = 5  # dimension of state

    # set maximal number of gradient steps
    maxGradSteps = 100

    # fix time parameters
    T = 1.  # terminal time
    steps = 10000  # number of time steps
    dt = T / steps  # time step size

    # initialize containers for ODE and referece solution
    steps = int(T / dt) + 1

    lam = 0.001  # cost parameter

    return noise_flag, M, N, maxGradSteps, T, steps, dt, lam


def initialize_input(M,T,t_vec,steps):
    # set input exponential chirp
    f0 = 3.
    f1 = 20.
    u = np.zeros([steps + 1, M])
    u[:, 0] = scs.chirp(t_vec, f0, T, f1)
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)
    # u[:,0] = scs.square(t_vec,0.1)

    # f0 = 2.
    # f1 = 25.
    # u[:,1] = scs.chirp(t_vec, f0,T,f1)

    # u[:,1] = scs.square(t_vec,0.1)
    u[:, 1] = np.sin(2. * np.pi * f0 * np.linspace(0, T, steps + 1))

    plt.figure(4)
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$u(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('input_det.png', dpi=300)

    return u


def init_B(N,M):

    B = np.zeros([N,M])
    B[:M,:M] = np.diag([4,2])
    B = B - 1+2*np.random.rand(N,M)

    return B

def initialize_reference_data(N,M):
    np.random.seed(10)
    # set true values to be identified later
    BT = init_B(N,M) #np.dot(A, A.transpose())

    QT = 1.1*np.eye(N) #np.diag([1.2,1.3,1.1,1.,1.2])
    JT = 10*generate_skew_symm(N)

    AR = -1 + 2*np.random.rand(N,N)
    RT = np.dot(AR, AR.transpose())
    #RT = 1./N*np.diag(range(1,N+1))
    RT[:2,:2] = 0.0
    #RT = remove_1_percent(RT)
    xT = np.ones([N,])
    return BT, QT, JT, RT, xT

def initialize_for_optimization(N,M):
    B = np.zeros([N,M])
    B[:M,:M] = np.diag([1.,3.])
    #B = init_B(N,M)
    #B = np.random.rand(N,M)
    Q = np.eye(N)
    J = 10*generate_skew_symm(N)
    AR = -1 + 2*np.random.rand(N, N)
    R = np.dot(AR, AR.transpose())# np.eye(N)
    R[:2,:2]=0.0
    x0 = 1.2*np.ones([N,])  #1./N*np.arange(N) #np.zeros([N,]) # np.ones([N,])
    return B, Q, J, R, x0

def initialize_directions(R,N,M):

    HB = np.zeros([N,M])
    HB[:M,:M] = np.diag([1,-1])
    Hx = np.ones([N,])
    HR = generate_symm(N)
    HQ = generate_symm(N)
    HJ = generate_skew_symm(N)

    return HB, HQ, HJ, HR, HR, Hx


############################################
#### functions for test with lambda > 0 ####
############################################
def initialize_solver_parameters_lam(steps,N,T):

    # choose if alternative retraction for Q is used
    alternative = False  # True /False

    # armijo parameter
    gamma = 0.001
    sig = 100.
    fail_flag = False
    eps = 0.000001

    # ODE solution
    sol_x = np.zeros((steps + 1, N))

    # reference solution
    ref_x = np.zeros((steps + 1, N))

    # time vec
    t_vec = np.linspace(0, T, steps + 1)

    return alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec

def initialilze_problem_parameters_lam(noiseFlag=False):
    # the determinisitic version runs without noise
    noise_flag = noiseFlag  # True/False

    # fix dimension of the matrices
    M = 2  # dimension of control
    N = 5  # dimension of state

    # set maximal number of gradient steps
    maxGradSteps = 100

    # fix time parameters
    T = 1.  # terminal time
    steps = 10000  # number of time steps
    dt = T / steps  # time step size

    # initialize containers for ODE and referece solution
    steps = int(T / dt) + 1

    lam = 0.001  # cost parameter

    return noise_flag, M, N, maxGradSteps, T, steps, dt, lam

def initialize_input_lam(M,T,t_vec,steps):
    # set input exponential chirp
    f0 = 3.
    f1 = 20.
    u = np.zeros([steps + 1, M])
    u[:, 0] = scs.chirp(t_vec, f0, T, f1)
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)
    # u[:,0] = scs.square(t_vec,0.1)

    # f0 = 2.
    # f1 = 25.
    # u[:,1] = scs.chirp(t_vec, f0,T,f1)

    # u[:,1] = scs.square(t_vec,0.1)
    u[:, 1] = np.sin(2. * np.pi * f0 * np.linspace(0, T, steps + 1))

    plt.figure(4)
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$u(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('input_det_lam.png', dpi=300)

    return u


def init_B_lam(N,M):

    B = np.zeros([N,M])
    B[:M,:M] = np.diag([4,2])
    B = B - 0.1+0.2*np.random.rand(N,M)

    return B

def initialize_reference_data_lam(N,M):

    # set true values to be identified later
    BT = init_B_lam(N,M)
    QT = np.diag(range(1,N+1))
    JT = generate_skew_symm(N)
    AR = -0.1 + 0.2*np.random.rand(N,N)
    RT = np.dot(AR, AR.transpose())
    RT[:2,:2] = 0.0 # semi-definite
    xT = np.zeros([N,])
    return BT, QT, JT, RT, xT

def initialize_for_optimization_lam(N,M):

    B = np.zeros([N,M])
    B[:M,:M] = np.diag([1,3])
    Q = np.eye(N)
    J = generate_skew_symm(N)
    AR = -0.1 + 0.2*np.random.rand(N, N)
    R = np.eye(N)
    R[:2,:2]=0.0 # semi-definite
    x0 = np.zeros([N,])
    return B, Q, J, R, x0

def plot_solution_initial_setting_lam(t_vec,yn):

    plt.figure(1)
    plt.plot(t_vec, yn[:, 0])
    plt.plot(t_vec, yn[:, 1])
    plt.xlabel(r'$t$', fontsize = 15)
    plt.ylabel(r'$y_\mathrm{initial}$', fontsize = 15)
    plt.tight_layout()
    plt.savefig('initialoutput_det_lam.png', dpi=300)
    plt.clf()

    plt.figure(8)
    plt.plot(t_vec,yn[:,0], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')
    plt.figure(9)
    plt.plot(t_vec,yn[:,1], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')

def armijo_rule_lam(sol_x,sol_y,J,R,Q,B,x0,XI_J,XI_R,XI_Q,XI_B,XI_x,ydata,u,sig,gamma,fail_flag,eps,alternative,dt,steps,N,lam):

    # updates
    Jn  = compute_retraction_J(J,sig*XI_J)
    Rn  = compute_retraction_R(R,sig*XI_R)
    Qn  = compute_retraction_Q(Q,sig*XI_Q)
    Bn  = compute_retraction_B(B,sig*XI_B)
    x0n = x0#compute_retraction_x(x0,sig*XI_x)

    c0 = evaluate_cost(sol_y, ydata, lam, Q, dt)

    xn, yn = solve_forward(x0n,Jn,Rn,Qn,Bn,u,steps,N,dt)

    cn = evaluate_cost(yn, ydata, lam, Qn, dt)


    scalar_J = -np.sum(np.power(J, 2))
    scalar_R = -np.sum(np.power(R, 2))
    # Q pos def
    uSVD, sSVD, vhSVD = np.linalg.svd(spl.logm(Q), full_matrices=True)
    scalar_Q = -sSVD[0]
    scalar_B = -np.sum(np.power(B, 2))
    scalar_x = -np.linalg.norm(x0)

    scalar = max(scalar_J,scalar_R,scalar_Q,scalar_B,scalar_x)



    while (cn-c0 > gamma*sig*scalar and sig > eps):
        sig = 0.5*sig

        Jn  = compute_retraction_J(J,  sig * XI_J)
        Rn  = compute_retraction_R(R,  sig * XI_R)
        Qn  = compute_retraction_Q(Q,  sig * XI_Q)
        Bn  = compute_retraction_B(B,  sig * XI_B)
        x0n = x0#compute_retraction_x(x0, sig * XI_x)

        xn, yn = solve_forward(x0n, Jn, Rn, Qn, Bn, u,steps,N,dt)

        cn = evaluate_cost(yn, ydata, lam, Qn, dt)

    if sig <= eps:
        print('Amijo failed')
        fail_flag = True

    return Jn, Rn, Qn, Bn, x0n, xn, yn, fail_flag, cn

def plots_postprocessing_lam(cost_vec, t_vec, yn, ydata, norm_vec_B, norm_vec_Q, norm_vec_J, norm_vec_R, i):
    cFirst = cost_vec[0]
    cLast = cost_vec[-1]
    print('First cost', cFirst)
    print('Last cost', cLast)
    print('last cost is ', cLast / cFirst * 100, '% of first cost')

    plt.figure(1)
    ax = plt.figure().gca()
    ax.set_xticks(np.arange(0, i + 1, 5,  dtype=int))
    plt.semilogy(cost_vec)
    plt.xlabel('iteration', fontsize = 13)
    plt.ylabel(r'$\hat\mathcal J$', fontsize=13)
    plt.tight_layout()
    plt.savefig('cost_det_lam.png', dpi=300)

    plt.figure(8)
    plt.plot(t_vec, yn[:, 0], label='fitted')
    plt.plot(t_vec, ydata[:, 0], '--', label='reference')
    plt.ylabel('output, first component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_first_lam.png', dpi=300)

    plt.figure(9)
    plt.plot(t_vec, yn[:, 1], label='fitted')
    plt.plot(t_vec, ydata[:, 1], '--', label='reference')
    plt.ylabel('output, second component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_second_lam.png', dpi=300)

def input_robustness_test_lam(t_vec,T,steps,M):

    u = u = np.zeros([steps + 1, M])

    # simple
    f0 = 1.
    f1 = 15.
    u[:,0] = scs.chirp(t_vec, f0,T,f1)
    u[:, 1] = np.cos(1. * np.pi * f0 * np.linspace(0, T, steps + 1))
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)

    # difficult
    u[:,0] = scs.square(t_vec,0.1)
    u[:,1] =np.cos(1.*np.pi*f0*np.linspace(0,T,steps+1))

    plt.figure(4)
    plt.clf()
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$', fontsize=15)
    plt.ylabel(r'$u_\mathrm{test}(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('test_input_det_lam.png', dpi=300)

    return u

def run_robustness_test_lam(xT,JT,RT,QT,BT,u,x0,J,R,Q,B,steps,N,dt):
    Rtest_ref_x, Rtest_ref_y = solve_forward(xT,JT,RT,QT,BT,u,steps,N,dt)
    Rtest_x, Rtest_y = solve_forward(x0,J,R,Q,B,u,steps,N,dt)

    return Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y

def plot_robustness_test_lam(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt):
    plt.figure(12)
    #plt.title('cross validation',fontsize=15)
    plt.ylabel('output, first component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,0], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 0], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_first_lam.png', dpi=300)

    plt.figure(13)
    #plt.title('Robustness y[:,1]', fontsize=15)
    plt.ylabel('output, second component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,1], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 1], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_second_lam.png', dpi=300)

    print('lam L2 norm diff y', np.sqrt(np.sum(np.power(Rtest_ref_y - Rtest_y,2)*dt)))
    print('lam max norm diff y', np.max(np.abs(Rtest_ref_y - Rtest_y)))
    
############################################
#### functions for test x_0 ####
############################################
def initialize_solver_parameters_X(steps,N,T):

    # choose if alternative retraction for Q is used
    alternative = False  # True /False

    # armijo parameter
    gamma = 0.001
    sig = 1000.
    fail_flag = False
    eps = 0.00000001

    # ODE solution
    sol_x = np.zeros((steps + 1, N))

    # reference solution
    ref_x = np.zeros((steps + 1, N))

    # time vec
    t_vec = np.linspace(0, T, steps + 1)

    return alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec

def initialilze_problem_parameters_X(noiseFlag=False):
    # the determinisitic version runs without noise
    noise_flag = noiseFlag  # True/False

    # fix dimension of the matrices
    M = 2  # dimension of control
    N = 5  # dimension of state

    # set maximal number of gradient steps
    maxGradSteps = 100

    # fix time parameters
    T = 1.  # terminal time
    steps = 10000  # number of time steps
    dt = T / steps  # time step size

    # initialize containers for ODE and referece solution
    steps = int(T / dt) + 1

    lam = 0.0  # cost parameter

    return noise_flag, M, N, maxGradSteps, T, steps, dt, lam

def initialize_input_X(M,T,t_vec,steps):
    # set input exponential chirp
    f0 = 3.
    f1 = 20.
    u = np.zeros([steps + 1, M])
    u[:, 0] = scs.chirp(t_vec, f0, T, f1)
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)
    # u[:,0] = scs.square(t_vec,0.1)

    # f0 = 2.
    # f1 = 25.
    # u[:,1] = scs.chirp(t_vec, f0,T,f1)

    # u[:,1] = scs.square(t_vec,0.1)
    u[:, 1] = np.sin(2. * np.pi * f0 * np.linspace(0, T, steps + 1))

    plt.figure(4)
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$u(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('input_detX.png', dpi=300)

    return u


def init_B_X(N,M):

    B = np.zeros([N,M])
    B[:M,:M] = np.diag([4,2])
    B = B - 0.1+0.2*np.random.rand(N,M)

    return B

def initialize_reference_data_X(N,M):
    # set true values to be identified later
    BT = init_B_X(N,M)
    QT = np.diag(range(1,N+1))
    JT = generate_skew_symm(N)
    AR = -0.1 + 0.2*np.random.rand(N,N)
    RT = np.dot(AR, AR.transpose())
    RT[:2,:2] = 0.0 # semi-definite
    xT = 0.1*np.arange(N)
    return BT, QT, JT, RT, xT

def initialize_for_optimization_X(N,M):

    x0 = np.zeros([N,])
    return x0

def plot_solution_initial_setting_X(t_vec,yn):

    plt.figure(1)
    plt.plot(t_vec, yn[:, 0])
    plt.plot(t_vec, yn[:, 1])
    plt.xlabel(r'$t$', fontsize = 15)
    plt.ylabel(r'$y_\mathrm{initial}$', fontsize = 15)
    plt.tight_layout()
    plt.savefig('initialoutput_detX.png', dpi=300)
    plt.clf()

    plt.figure(8)
    plt.plot(t_vec,yn[:,0], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')
    plt.figure(9)
    plt.plot(t_vec,yn[:,1], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')

def armijo_rule_X(sol_x,sol_y,J,R,Q,B,x0,XI_J,XI_R,XI_Q,XI_B,XI_x,ydata,u,sig,gamma,fail_flag,eps,alternative,dt,steps,N,lam):

    # updates
    Jn  = J#compute_retraction_J(J,sig*XI_J)
    Rn  = R#compute_retraction_R(R,sig*XI_R)
    Qn  = Q#compute_retraction_Q(Q,sig*XI_Q,alternative)
    Bn  = B#compute_retraction_B(B,sig*XI_B)
    x0n = compute_retraction_x(x0,sig*XI_x)

    c0 = evaluate_cost(sol_y,ydata,lam,Q,dt)

    xn, yn = solve_forward(x0n,Jn,Rn,Qn,Bn,u,steps,N,dt)

    cn = evaluate_cost(yn, ydata,lam, Qn, dt)


    scalar_J = -np.sum(np.power(J, 2))
    scalar_R = -np.sum(np.power(R, 2))
    # Q pos def
    uSVD, sSVD, vhSVD = np.linalg.svd(spl.logm(Q), full_matrices=True)
    scalar_Q = -sSVD[0]
    scalar_B = -np.sum(np.power(B, 2))
    scalar_x = -np.linalg.norm(x0)

    scalar = max(scalar_J,scalar_R,scalar_Q,scalar_B,scalar_x)



    while (cn-c0 > gamma*sig*scalar and sig > eps):
        sig = 0.5*sig

        Jn  = J#compute_retraction_J(J,  sig * XI_J)
        Rn  = R#compute_retraction_R(R,  sig * XI_R)
        Qn  = Q#compute_retraction_Q(Q,  sig * XI_Q, alternative)
        Bn  = B#compute_retraction_B(B,  sig * XI_B)
        x0n = compute_retraction_x(x0, sig * XI_x)

        xn, yn = solve_forward(x0n, Jn, Rn, Qn, Bn, u,steps,N,dt)

        cn = evaluate_cost(yn, ydata, lam, Qn, dt)

    if sig < eps:
        fail_flag = True

    return Jn, Rn, Qn, Bn, x0n, xn, yn, fail_flag, cn


def plots_postprocessing_X(cost_vec, t_vec, yn, ydata, norm_vec_B, norm_vec_Q, norm_vec_J, norm_vec_R, i):
    cFirst = cost_vec[0]
    cLast = cost_vec[-1]
    print('First cost', cFirst)
    print('Last cost', cLast)
    print('last cost is ', cLast / cFirst * 100, '% of first cost')

    plt.figure(1)
    ax = plt.figure().gca()
    ax.set_xticks(np.arange(0, i + 1, 5, dtype=int))
    plt.semilogy(cost_vec)
    plt.xlabel('iteration', fontsize=13)
    plt.ylabel(r'$\hat\mathcal J$', fontsize=13)
    plt.tight_layout()
    plt.savefig('cost_detX.png', dpi=300)

    plt.figure(8)
    plt.plot(t_vec, yn[:, 0], label='fitted')
    plt.plot(t_vec, ydata[:, 0], '--', label='reference')
    plt.ylabel('output, first component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_firstX.png', dpi=300)

    plt.figure(9)
    plt.plot(t_vec, yn[:, 1], label='fitted')
    plt.plot(t_vec, ydata[:, 1], '--', label='reference')
    plt.ylabel('output, second component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_secondX.png', dpi=300)


def input_robustness_test_X(t_vec, T, steps, M):
    u = u = np.zeros([steps + 1, M])

    # simple
    f0 = 1.
    f1 = 15.
    u[:, 0] = scs.chirp(t_vec, f0, T, f1)
    u[:, 1] = np.cos(1. * np.pi * f0 * np.linspace(0, T, steps + 1))
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)

    # difficult
    u[:, 0] = scs.square(t_vec, 0.1)
    u[:, 1] = np.cos(1. * np.pi * f0 * np.linspace(0, T, steps + 1))

    plt.figure(4)
    plt.clf()
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$', fontsize=15)
    plt.ylabel(r'$u_\mathrm{test}(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('test_input_detX.png', dpi=300)

    return u


def run_robustness_test_X(xT, JT, RT, QT, BT, u, x0, J, R, Q, B, steps, N, dt):
    Rtest_ref_x, Rtest_ref_y = solve_forward(xT, JT, RT, QT, BT, u, steps, N, dt)
    Rtest_x, Rtest_y = solve_forward(x0, J, R, Q, B, u, steps, N, dt)

    return Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y


def plot_robustness_test_X(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt):
    plt.figure(12)
    # plt.title('cross validation',fontsize=15)
    plt.ylabel('output, first component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:, 0], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 0], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_firstX.png', dpi=300)

    plt.figure(13)
    # plt.title('Robustness y[:,1]', fontsize=15)
    plt.ylabel('output, second component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:, 1], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 1], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_secondX.png', dpi=300)

    print('L2 norm diff y', np.sqrt(np.sum(np.power(Rtest_ref_y - Rtest_y, 2) * dt)))
    print('max norm diff y', np.max(np.abs(Rtest_ref_y - Rtest_y)))


############################################
#### functions for test with noise ####
############################################
def initialize_solver_parameters_sto(steps,N,T):

    # choose if alternative retraction for Q is used
    alternative = False  # True /False

    # armijo parameter
    gamma = 0.001
    sig = 1000.
    fail_flag = False
    eps = 0.00000001

    # ODE solution
    sol_x = np.zeros((steps + 1, N))

    # reference solution
    ref_x = np.zeros((steps + 1, N))

    # time vec
    t_vec = np.linspace(0, T, steps + 1)

    return alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec

def initialilze_problem_parameters_sto(noiseFlag=False):
    # the determinisitic version runs without noise
    noise_flag = noiseFlag  # True/False

    # fix dimension of the matrices
    M = 2  # dimension of control
    N = 5  # dimension of state

    # set maximal number of gradient steps
    maxGradSteps = 100

    # fix time parameters
    T = 1.  # terminal time
    steps = 10000  # number of time steps
    dt = T / steps  # time step size

    # initialize containers for ODE and referece solution
    steps = int(T / dt) + 1

    lam = 0.0  # cost parameter

    return noise_flag, M, N, maxGradSteps, T, steps, dt, lam

def initialize_input_sto(M,T,t_vec,steps):
    # set input exponential chirp
    f0 = 3.
    f1 = 20.
    u = np.zeros([steps + 1, M])
    u[:, 0] = scs.chirp(t_vec, f0, T, f1)
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)
    # u[:,0] = scs.square(t_vec,0.1)

    # f0 = 2.
    # f1 = 25.
    # u[:,1] = scs.chirp(t_vec, f0,T,f1)

    # u[:,1] = scs.square(t_vec,0.1)
    u[:, 1] = np.sin(2. * np.pi * f0 * np.linspace(0, T, steps + 1))

    plt.figure(4)
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$u(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('input_det.png', dpi=300)

    return u


def init_B_sto(N,M):

    B = np.zeros([N,M])
    B[:M,:M] = np.diag([4,2])
    B = B - 0.1+0.2*np.random.rand(N,M)

    return B

def initialize_reference_data_sto(N,M):
    # set true values to be identified later
    BT = init_B_sto(N,M)
    QT = np.diag(range(1,N+1))
    JT = generate_skew_symm(N)
    AR = -0.1 + 0.2*np.random.rand(N,N)
    RT = np.dot(AR, AR.transpose())
    RT[:2,:2] = 0.0 # semi-definite
    xT = np.zeros([N,])
    return BT, QT, JT, RT, xT

def initialize_for_optimization_sto(N,M):
    B = np.zeros([N,M])
    B[:M,:M] = np.diag([1,3])
    Q = np.eye(N)
    J = generate_skew_symm(N)
    AR = -0.1 + 0.2*np.random.rand(N, N)
    R = np.eye(N)
    R[:2,:2]=0.0 # semi-definite
    x0 = np.zeros([N,])
    return B, Q, J, R, x0

def plot_solution_initial_setting_sto(t_vec,yn):

    plt.figure(1)
    plt.plot(t_vec, yn[:, 0])
    plt.plot(t_vec, yn[:, 1])
    plt.xlabel(r'$t$', fontsize = 15)
    plt.ylabel(r'$y_\mathrm{initial}$', fontsize = 15)
    plt.tight_layout()
    plt.savefig('initialoutput_sto.png', dpi=300)
    plt.clf()

    plt.figure(8)
    plt.plot(t_vec,yn[:,0], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')
    plt.figure(9)
    plt.plot(t_vec,yn[:,1], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')

def armijo_rule_sto(sol_x,sol_y,J,R,Q,B,x0,XI_J,XI_R,XI_Q,XI_B,XI_x,ydata,u,sig,gamma,fail_flag,eps,alternative,dt,steps,N,lam):

    # updates
    Jn  = compute_retraction_J(J,sig*XI_J)
    Rn  = compute_retraction_R(R,sig*XI_R)
    Qn  = Q#compute_retraction_Q(Q,sig*XI_Q,alternative)
    Bn  = compute_retraction_B(B,sig*XI_B)
    x0n = x0#compute_retraction_x(x0,sig*XI_x)

    c0 = evaluate_cost(sol_y, ydata, lam, Q, dt)

    xn, yn = solve_forward(x0n,Jn,Rn,Qn,Bn,u,steps,N,dt)

    cn = evaluate_cost(yn, ydata, lam, Qn, dt)


    scalar_J = -np.sum(np.power(J, 2))
    scalar_R = -np.sum(np.power(R, 2))
    # Q pos def
    uSVD, sSVD, vhSVD = np.linalg.svd(spl.logm(Q), full_matrices=True)
    scalar_Q = -sSVD[0]
    scalar_B = -np.sum(np.power(B, 2))
    scalar_x = -np.linalg.norm(x0)

    scalar = max(scalar_J,scalar_R,scalar_Q,scalar_B,scalar_x)



    while (cn-c0 > gamma*sig*scalar and sig > eps):
        sig = 0.5*sig

        Jn  = compute_retraction_J(J,  sig * XI_J)
        Rn  = compute_retraction_R(R,  sig * XI_R)
        Qn  = Q#compute_retraction_Q(Q,  sig * XI_Q, alternative)
        Bn  = compute_retraction_B(B,  sig * XI_B)
        x0n = x0#compute_retraction_x(x0, sig * XI_x)

        xn, yn = solve_forward(x0n, Jn, Rn, Qn, Bn, u,steps,N,dt)

        cn = evaluate_cost(yn, ydata, lam, Qn, dt)

    if sig < eps:
        fail_flag = True

    return Jn, Rn, Qn, Bn, x0n, xn, yn, fail_flag, cn

def plots_postprocessing_sto(cost_vec, t_vec, yn, ydata, norm_vec_B, norm_vec_Q, norm_vec_J, norm_vec_R, i):
    cFirst = cost_vec[0]
    cLast = cost_vec[-1]
    print('First cost', cFirst)
    print('Last cost', cLast)
    print('last cost is ', cLast / cFirst * 100, '% of first cost')

    plt.figure(1)
    ax = plt.figure().gca()
    ax.set_xticks(np.arange(0, i + 1, 5,  dtype=int))
    plt.semilogy(cost_vec)
    plt.xlabel('iteration', fontsize = 13)
    plt.ylabel(r'$\hat\mathcal J$', fontsize=13)
    plt.tight_layout()
    plt.savefig('cost_sto.png', dpi=300)

    plt.figure(8)
    plt.plot(t_vec, yn[:, 0], label='fitted')
    plt.plot(t_vec, ydata[:, 0], '--', label='reference')
    plt.ylabel('output, first component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_first_sto.png', dpi=300)

    plt.figure(9)
    plt.plot(t_vec, yn[:, 1], label='fitted')
    plt.plot(t_vec, ydata[:, 1], '--', label='reference')
    plt.ylabel('output, second component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_second_sto.png', dpi=300)

def input_robustness_test_sto(t_vec,T,steps,M):

    u = u = np.zeros([steps + 1, M])

    # simple
    f0 = 1.
    f1 = 15.
    u[:,0] = scs.chirp(t_vec, f0,T,f1)
    u[:, 1] = np.cos(1. * np.pi * f0 * np.linspace(0, T, steps + 1))
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)

    # difficult
    u[:,0] = scs.square(t_vec,0.1)
    u[:,1] =np.cos(1.*np.pi*f0*np.linspace(0,T,steps+1))

    plt.figure(4)
    plt.clf()
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$', fontsize=15)
    plt.ylabel(r'$u_\mathrm{test}(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('test_input_sto.png', dpi=300)

    return u

def run_robustness_test_sto(xT,JT,RT,QT,BT,u,x0,J,R,Q,B,steps,N,dt):
    Rtest_ref_x, Rtest_ref_y = solve_forward(xT,JT,RT,QT,BT,u,steps,N,dt)
    Rtest_x, Rtest_y = solve_forward(x0,J,R,Q,B,u,steps,N,dt)

    return Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y

def plot_robustness_test_sto(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt):
    plt.figure(12)
    #plt.title('cross validation',fontsize=15)
    plt.ylabel('output, first component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,0], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 0], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_first_sto.png', dpi=300)

    plt.figure(13)
    #plt.title('Robustness y[:,1]', fontsize=15)
    plt.ylabel('output, second component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,1], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 1], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_second_sto.png', dpi=300)

    print('L2 norm diff y', np.sqrt(np.sum(np.power(Rtest_ref_y - Rtest_y,2)*dt)))
    print('max norm diff y', np.max(np.abs(Rtest_ref_y - Rtest_y)))



############################################
#### functions for deterministic test ####
############################################
def initialize_solver_parameters_det(steps,N,T):

    # choose if alternative retraction for Q is used
    alternative = False  # True /False

    # armijo parameter
    gamma = 0.001
    sig = 1000.
    fail_flag = False
    eps = 0.00000001

    # ODE solution
    sol_x = np.zeros((steps + 1, N))

    # reference solution
    ref_x = np.zeros((steps + 1, N))

    # time vec
    t_vec = np.linspace(0, T, steps + 1)

    return alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec

def initialilze_problem_parameters_det(noiseFlag=False):
    # the determinisitic version runs without noise
    noise_flag = noiseFlag  # True/False

    # fix dimension of the matrices
    M = 2  # dimension of control
    N = 5  # dimension of state

    # set maximal number of gradient steps
    maxGradSteps = 100

    # fix time parameters
    T = 1.  # terminal time
    steps = 10000  # number of time steps
    dt = T / steps  # time step size

    # initialize containers for ODE and referece solution
    steps = int(T / dt) + 1

    lam = 0.0  # cost parameter

    return noise_flag, M, N, maxGradSteps, T, steps, dt, lam

def initialize_input_det(M,T,t_vec,steps):
    # set input exponential chirp
    f0 = 3.
    f1 = 20.
    u = np.zeros([steps + 1, M])
    u[:, 0] = scs.chirp(t_vec, f0, T, f1)
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)
    # u[:,0] = scs.square(t_vec,0.1)

    # f0 = 2.
    # f1 = 25.
    # u[:,1] = scs.chirp(t_vec, f0,T,f1)

    # u[:,1] = scs.square(t_vec,0.1)
    u[:, 1] = np.sin(2. * np.pi * f0 * np.linspace(0, T, steps + 1))

    plt.figure(4)
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$u(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('input_det.png', dpi=300)

    return u


def init_B_det(N,M):

    B = np.zeros([N,M])
    B[:M,:M] = np.diag([4,2])
    B = B - 0.1+0.2*np.random.rand(N,M)

    return B

def initialize_reference_data_det(N,M):
    # set true values to be identified later
    BT = init_B_det(N,M)
    QT = np.diag(range(1,N+1))
    JT = generate_skew_symm(N)
    AR = -0.1 + 0.2*np.random.rand(N,N)
    RT = np.dot(AR, AR.transpose())
    RT[:2,:2] = 0.0 # semi-definite
    xT = np.zeros([N,])
    return BT, QT, JT, RT, xT

def initialize_for_optimization_det(N,M):
    B = np.zeros([N,M])
    B[:M,:M] = np.diag([1,3])
    Q = np.eye(N)
    J = generate_skew_symm(N)
    AR = -0.1 + 0.2*np.random.rand(N, N)
    R = np.eye(N)
    R[:2,:2]=0.0 # semi-definite
    x0 = np.zeros([N,])
    return B, Q, J, R, x0

def plot_solution_initial_setting_det(t_vec,yn):

    plt.figure(1)
    plt.plot(t_vec, yn[:, 0])
    plt.plot(t_vec, yn[:, 1])
    plt.xlabel(r'$t$', fontsize = 15)
    plt.ylabel(r'$y_\mathrm{initial}$', fontsize = 15)
    plt.tight_layout()
    plt.savefig('initialoutput_det.png', dpi=300)
    plt.clf()

    plt.figure(8)
    plt.plot(t_vec,yn[:,0], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')
    plt.figure(9)
    plt.plot(t_vec,yn[:,1], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')

def armijo_rule_det(sol_x,sol_y,J,R,Q,B,x0,XI_J,XI_R,XI_Q,XI_B,XI_x,ydata,u,sig,gamma,fail_flag,eps,alternative,dt,steps,N,lam):

    # updates
    Jn  = compute_retraction_J(J,sig*XI_J)
    Rn  = compute_retraction_R(R,sig*XI_R)
    Qn  = Q#compute_retraction_Q(Q,sig*XI_Q,alternative)
    Bn  = compute_retraction_B(B,sig*XI_B)
    x0n = x0#compute_retraction_x(x0,sig*XI_x)

    c0 = evaluate_cost(sol_y, ydata, lam, Q, dt)

    xn, yn = solve_forward(x0n,Jn,Rn,Qn,Bn,u,steps,N,dt)

    cn = evaluate_cost(yn, ydata, lam, Qn, dt)


    scalar_J = -np.sum(np.power(J, 2))
    scalar_R = -np.sum(np.power(R, 2))
    # Q pos def
    uSVD, sSVD, vhSVD = np.linalg.svd(spl.logm(Q), full_matrices=True)
    scalar_Q = -sSVD[0]
    scalar_B = -np.sum(np.power(B, 2))
    scalar_x = -np.linalg.norm(x0)

    scalar = max(scalar_J,scalar_R,scalar_Q,scalar_B,scalar_x)



    while (cn-c0 > gamma*sig*scalar and sig > eps):
        sig = 0.5*sig

        Jn  = compute_retraction_J(J,  sig * XI_J)
        Rn  = compute_retraction_R(R,  sig * XI_R)
        Qn  = Q#compute_retraction_Q(Q,  sig * XI_Q, alternative)
        Bn  = compute_retraction_B(B,  sig * XI_B)
        x0n = x0#compute_retraction_x(x0, sig * XI_x)

        xn, yn = solve_forward(x0n, Jn, Rn, Qn, Bn, u,steps,N,dt)

        cn = evaluate_cost(yn, ydata,lam, Qn, dt)

    if sig < eps:
        fail_flag = True

    return Jn, Rn, Qn, Bn, x0n, xn, yn, fail_flag, cn

def plots_postprocessing_det(cost_vec, t_vec, yn, ydata, norm_vec_B, norm_vec_Q, norm_vec_J, norm_vec_R, i):
    cFirst = cost_vec[0]
    cLast = cost_vec[-1]
    print('First cost', cFirst)
    print('Last cost', cLast)
    print('last cost is ', cLast / cFirst * 100, '% of first cost')

    plt.figure(1)
    ax = plt.figure().gca()
    ax.set_xticks(np.arange(0, i + 1, 5,  dtype=int))
    plt.semilogy(cost_vec)
    plt.xlabel('iteration', fontsize = 13)
    plt.ylabel(r'$\hat\mathcal J$', fontsize=13)
    plt.tight_layout()
    plt.savefig('cost_det.png', dpi=300)

    plt.figure(8)
    plt.plot(t_vec, yn[:, 0], label='fitted')
    plt.plot(t_vec, ydata[:, 0], '--', label='reference')
    plt.ylabel('output, first component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_first.png', dpi=300)

    plt.figure(9)
    plt.plot(t_vec, yn[:, 1], label='fitted')
    plt.plot(t_vec, ydata[:, 1], '--', label='reference')
    plt.ylabel('output, second component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_second.png', dpi=300)

def input_robustness_test_det(t_vec,T,steps,M):

    u = u = np.zeros([steps + 1, M])

    # simple
    f0 = 1.
    f1 = 15.
    u[:,0] = scs.chirp(t_vec, f0,T,f1)
    u[:, 1] = np.cos(1. * np.pi * f0 * np.linspace(0, T, steps + 1))
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)

    # difficult
    u[:,0] = scs.square(t_vec,0.1)
    u[:,1] =np.cos(1.*np.pi*f0*np.linspace(0,T,steps+1))

    plt.figure(4)
    plt.clf()
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$', fontsize=15)
    plt.ylabel(r'$u_\mathrm{test}(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('test_input_det.png', dpi=300)

    return u

def run_robustness_test_det(xT,JT,RT,QT,BT,u,x0,J,R,Q,B,steps,N,dt):
    Rtest_ref_x, Rtest_ref_y = solve_forward(xT,JT,RT,QT,BT,u,steps,N,dt)
    Rtest_x, Rtest_y = solve_forward(x0,J,R,Q,B,u,steps,N,dt)

    return Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y

def plot_robustness_test_det(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt):
    plt.figure(12)
    #plt.title('cross validation',fontsize=15)
    plt.ylabel('output, first component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,0], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 0], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_first.png', dpi=300)

    plt.figure(13)
    #plt.title('Robustness y[:,1]', fontsize=15)
    plt.ylabel('output, second component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,1], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 1], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_second.png', dpi=300)

    print('L2 norm diff y', np.sqrt(np.sum(np.power(Rtest_ref_y - Rtest_y,2)*dt)))
    print('max norm diff y', np.max(np.abs(Rtest_ref_y - Rtest_y)))

############################################
#### functions for model reduction (MSD) ####
############################################

def initialize_solver_parameters_MSD(steps, N, T):
    # choose if alternative retraction for Q is used
    alternative = False  # True /False

    # armijo parameter
    gamma = 0.0001
    sig = 1000.
    fail_flag = False
    eps = 0.00000001

    # ODE solution
    sol_x = np.zeros((steps + 1, N))

    # reference solution
    ref_x = np.zeros((steps + 1, N))

    # time vec
    t_vec = np.linspace(0, T, steps + 1)

    return alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec

def initialilze_problem_parameters_MSD(m, n, noiseFlag=False):
    # the determinisitic version runs without noise
    noise_flag = noiseFlag  # True/False

    # fix dimension of the matrices
    M = m  # dimension of control
    N = n  # dimension of state

    # set maximal number of gradient steps
    maxGradSteps = 100

    # fix time parameters
    T = 1.  # terminal time
    steps = 10000  # number of time steps
    dt = T / steps  # time step size

    # initialize containers for ODE and referece solution
    steps = int(T / dt) + 1

    lam = 0.0  # cost parameter

    return noise_flag, M, N, maxGradSteps, T, steps, dt, lam

def initialize_input_MSD(M,T,t_vec,steps):
    # set input exponential chirp
    f0 = 3.
    f1 = 20.
    u = np.zeros([steps + 1, M])
    u[:, 0] = scs.chirp(t_vec, f0, T, f1)
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)
    # u[:,0] = scs.square(t_vec,0.1)

    # f0 = 2.
    # f1 = 25.
    # u[:,1] = scs.chirp(t_vec, f0,T,f1)

    # u[:,1] = scs.square(t_vec,0.1)
    u[:, 1] = np.sin(2. * np.pi * f0 * np.linspace(0, T, steps + 1))

    plt.figure(4)
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$u(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('input_det.png', dpi=300)

    return u


def init_B_MSD(N,M):

    B = np.zeros([N,M])
    B[:M,:M] = np.diag([4,2])
    B = B - 0.1+0.2*np.random.rand(N,M)

    return B

def initialize_reference_data_MSD():
    # set true values to be identified later
    BT = np.load('B.npy')
    QT = np.load('Q.npy')
    JT = np.load('J.npy')
    RT = np.load('R.npy')

    N,M = BT.shape
    xT = np.zeros([N,])
    return BT, QT, JT, RT, xT

def initialize_for_optimization_MSD(N,M):
    B = np.zeros([N,M])
    B[:M,:M] = np.diag([1,3])
    Q = np.eye(N)
    J = generate_skew_symm(N)
    R = np.eye(N)
    R[:int(0.5*N),:int(0.5*N)]=0.0 # semi-definite
    x0 = np.zeros([N,])
    return B, Q, J, R, x0

def plot_solution_initial_setting_MSD(t_vec,yn):

    plt.figure(1)
    plt.plot(t_vec, yn[:, 0])
    plt.plot(t_vec, yn[:, 1])
    plt.xlabel(r'$t$', fontsize = 15)
    plt.ylabel(r'$y_\mathrm{initial}$', fontsize = 15)
    plt.tight_layout()
    plt.savefig('initialoutput_MSD.png', dpi=300)
    plt.clf()

    plt.figure(8)
    plt.plot(t_vec,yn[:,0], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')
    plt.figure(9)
    plt.plot(t_vec,yn[:,1], label="inital")
    plt.xlabel('t')
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')

def armijo_rule_MSD(sol_x,sol_y,J,R,Q,B,x0,XI_J,XI_R,XI_Q,XI_B,XI_x,ydata,u,sig,gamma,fail_flag,eps,alternative,dt,steps,N,lam):

    # updates
    Jn  = compute_retraction_J(J,sig*XI_J)
    Rn  = compute_retraction_R(R,sig*XI_R)
    Qn  = Q#compute_retraction_Q(Q,sig*XI_Q,alternative)
    Bn  = compute_retraction_B(B,sig*XI_B)
    x0n = x0#compute_retraction_x(x0,sig*XI_x)

    c0 = evaluate_cost(sol_y, ydata,lam,Q,dt)

    xn, yn = solve_forward(x0n,Jn,Rn,Qn,Bn,u,steps,N,dt)

    cn = evaluate_cost(yn, ydata,lam,Qn, dt)


    scalar_J = -np.sum(np.power(J, 2))
    scalar_R = -np.sum(np.power(R, 2))
    # Q pos def
    uSVD, sSVD, vhSVD = np.linalg.svd(spl.logm(Q), full_matrices=True)
    scalar_Q = -sSVD[0]
    scalar_B = -np.sum(np.power(B, 2))
    scalar_x = -np.linalg.norm(x0)

    scalar = max(scalar_J,scalar_R,scalar_Q,scalar_B,scalar_x)



    while (cn-c0 > gamma*sig*scalar and sig > eps):
        sig = 0.5*sig

        Jn  = compute_retraction_J(J,  sig * XI_J)
        Rn  = compute_retraction_R(R,  sig * XI_R)
        Qn  = Q#compute_retraction_Q(Q,  sig * XI_Q, alternative)
        Bn  = compute_retraction_B(B,  sig * XI_B)
        x0n = x0#compute_retraction_x(x0, sig * XI_x)

        xn, yn = solve_forward(x0n, Jn, Rn, Qn, Bn, u,steps,N,dt)

        cn = evaluate_cost(yn, ydata,lam,Qn, dt)

    if sig < eps:
        fail_flag = True

    return Jn, Rn, Qn, Bn, x0n, xn, yn, fail_flag, cn

def plots_postprocessing_MSD(cost_vec, t_vec, yn, ydata, norm_vec_B, norm_vec_Q, norm_vec_J, norm_vec_R, i):
    cFirst = cost_vec[0]
    cLast = cost_vec[-1]
    print('First cost', cFirst)
    print('Last cost', cLast)
    print('last cost is ', cLast / cFirst * 100, '% of first cost')

    plt.figure(1)
    ax = plt.figure().gca()
    ax.set_xticks(np.arange(0, i + 1, 5,  dtype=int))
    plt.semilogy(cost_vec)
    plt.xlabel('iteration', fontsize = 13)
    plt.ylabel(r'$\hat\mathcal J$', fontsize=13)
    plt.tight_layout()
    plt.savefig('cost_MSD.png', dpi=300)

    plt.figure(8)
    plt.plot(t_vec, yn[:, 0], label='fitted')
    plt.plot(t_vec, ydata[:, 0], '--', label='reference')
    plt.ylabel('output, first component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_first_MSD.png', dpi=300)

    plt.figure(9)
    plt.plot(t_vec, yn[:, 1], label='fitted')
    plt.plot(t_vec, ydata[:, 1], '--', label='reference')
    plt.ylabel('output, second component')
    plt.xlabel(r'$t$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_second_MSD.png', dpi=300)

def input_robustness_test_MSD(t_vec,T,steps,M):

    u = u = np.zeros([steps + 1, M])

    # simple
    f0 = 1.
    f1 = 15.
    u[:,0] = scs.chirp(t_vec, f0,T,f1)
    u[:, 1] = np.cos(1. * np.pi * f0 * np.linspace(0, T, steps + 1))
    # u[:,0] = scs.gausspulse(t_vec-1., fc=5)
    # u[:,0] = scs.sawtooth(t_vec,0.01)

    # difficult
    u[:,0] = scs.square(t_vec,0.1)
    u[:,1] =np.cos(1.*np.pi*f0*np.linspace(0,T,steps+1))

    plt.figure(4)
    plt.clf()
    plt.plot(t_vec, u[:, 0])
    plt.plot(t_vec, u[:, 1])
    plt.xlabel(r'$t$', fontsize=15)
    plt.ylabel(r'$u_\mathrm{test}(t)$', fontsize=15)
    plt.tight_layout()
    plt.savefig('test_input_MSD.png', dpi=300)

    return u

def run_robustness_test_MSD(xT,JT,RT,QT,BT,u,x0,J,R,Q,B,steps,NT,N,dt):
    Rtest_ref_x, Rtest_ref_y = solve_forward(xT,JT,RT,QT,BT,u,steps,NT,dt)
    Rtest_x, Rtest_y = solve_forward(x0,J,R,Q,B,u,steps,N,dt)

    return Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y

def plot_robustness_test_MSD(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt):
    plt.figure(12)
    #plt.title('cross validation',fontsize=15)
    plt.ylabel('output, first component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,0], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 0], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_first_MSD.png', dpi=300)

    plt.figure(13)
    #plt.title('Robustness y[:,1]', fontsize=15)
    plt.ylabel('output, second component', fontsize=13)
    plt.xlabel(r'$t$', fontsize=13)
    plt.plot(t_vec, Rtest_y[:,1], label='fitted')
    plt.plot(t_vec, Rtest_ref_y[:, 1], '--', label='reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robustness_second_MSD.png', dpi=300)

    print('_MSD L2 norm diff y', np.sqrt(np.sum(np.power(Rtest_ref_y - Rtest_y,2)*dt)))
    print('_MSD max norm diff y', np.max(np.abs(Rtest_ref_y - Rtest_y)))

def save_initial_ref_setting_MSD(t_vec, ref_y, currN):
    np.save('y_initial_ref{0}.npy'.format(currN),ref_y)

def plot_and_save_solution_initial_setting_MSD(t_vec,yn,currN):
    np.save('y_initial{0}.npy'.format(currN), yn)

    plt.figure(1)
    plt.plot(t_vec, yn[:, 0], label='{0}'.format(currN))
    plt.plot(t_vec, yn[:, 1], label='{0}'.format(currN))
    plt.xlabel(r'$t$', fontsize = 15)
    plt.ylabel(r'$y_\mathrm{initial}$', fontsize = 15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('initialoutput_det.png', dpi=300)
    plt.clf()

    plt.figure(8)
    plt.plot(t_vec,yn[:,0], label='{0}'.format(currN))
    plt.xlabel('t')
    plt.legend()
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')
    plt.figure(9)
    plt.plot(t_vec,yn[:,1], label='{0}'.format(currN))
    plt.xlabel('t')
    plt.legend()
    #plt.ylabel(r'$y_\mathrm{initial}(t)$')
    # plt.savefig('initialoutput_det.png')



def plot_model_reduction_MSD(t_vec,N_vec):

    plt.close('all')

    ydiff = []
    ydiff_rel = []
    ydiff_rel_initial = []

    for currN in N_vec:
        c = np.load('cost_MSD{0}.npy'.format(currN))
        y = np.load('ydiffmax_MSD{0}.npy'.format(currN))
        yi = np.load('y_initial{0}.npy'.format(currN))
        yr = np.load('y_initial_ref{0}.npy'.format(currN))
        yn = np.load('yopt_MSD{0}.npy'.format(currN))

        yrmax = np.max(np.abs(yr))
        yimax = np.max(np.abs(yi-yr))

        ydiff.append(y)
        ydiff_rel.append(y/yrmax)
        ydiff_rel_initial.append(yimax/yrmax)

        plt.figure(1)
        plt.semilogy(c[:100], label='{0}'.format(currN))

        plt.figure(4)
        plt.plot(t_vec,yi[:,0],  label='{0}'.format(currN))

        plt.figure(5)
        plt.plot(t_vec,yi[:,1], label='{0}'.format(currN))

        plt.figure(6)
        plt.plot(t_vec,yn[:, 0], label='{0}'.format(currN))

        plt.figure(7)
        plt.plot(t_vec,yn[:, 1], label='{0}'.format(currN))


    plt.figure(1)
    plt.xlabel('iterations', fontsize=13)
    plt.ylabel(r'$\hat\mathcal J$', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('MSD_cost.png', dpi=300)

    plt.figure(2)
    plt.plot(N_vec, ydiff)
    plt.xlabel(r'$n$', fontsize=13)
    plt.ylabel(r'$\| y_n - y_\mathrm{ref}\|_\infty$', fontsize=13)
    plt.tight_layout()
    plt.savefig('MSD_ydiff.png', dpi=300)

    plt.figure(3)
    plt.semilogy(N_vec, ydiff_rel, label = 'fitted')
    plt.semilogy(N_vec, ydiff_rel_initial, label = 'initial')
    plt.xlabel(r'$n$', fontsize=13)
    plt.ylabel(r'$\| y_n - y_\mathrm{ref}\|_\infty / \|y_\mathrm{ref}\|_\infty$', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('MSD_ydiff_rel.png', dpi=300)

    plt.figure(4)
    plt.plot(t_vec,yr[:, 0], 'k--', label='ref')
    plt.xlabel('t')
    plt.ylabel(r'$y_\mathrm{initial}$, first component', fontsize = 13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('initialoutput_first_MSD.png', dpi=300)

    plt.figure(5)
    plt.plot(t_vec,yr[:, 1], 'k--', label='ref')
    plt.xlabel('t')
    plt.ylabel(r'$y_\mathrm{initial}$, second component', fontsize = 13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('initialoutput_second_MSD.png', dpi=300)

    plt.figure(6)
    plt.plot(t_vec,yr[:, 0], 'k--', label='ref')
    plt.xlabel('t')
    plt.ylabel(r'$y_\mathrm{fitted}$, first component', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fitted_output_first_MSD.png', dpi=300)

    plt.figure(7)
    plt.plot(t_vec,yr[:, 1], 'k--', label='ref')
    plt.xlabel('t')
    plt.ylabel(r'$y_\mathrm{fitted}$, second component', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fitted_output_second_MSD.png', dpi=300)


    ######
    print('y_diff_rel', ydiff_rel)
    print('y_diff_rel_initial', ydiff_rel_initial)

    # show plots
    plt.show()
