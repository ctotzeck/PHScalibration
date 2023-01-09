''' Code written by C. Totzeck, Dec 2022

    Run the code to obtain the data and results of the first test case presented in the article
    'Data-driven adjoint-based calibration of port-Hamiltonian systems in time domain'
    by Michael Günther, Birgit Jacob and Claudia Totzeck (IMACM, University of Wuppertal)

'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl
import copy

np.random.seed(10)

plt.rcParams['text.usetex'] = True

noise_flag = False #True # True/False

maxGradSteps = 100

M = 2 # dimension of control
N = 20 # dimension of state

alternative = False #True #False #True

T = 1.
steps = 1000
dt = T/steps

lam = 0.0    # cost parameter
gamma = 0.00 # armijo parameter

cost_vec = []
norm_vec_Q = []
norm_vec_J = []
norm_vec_R = []
norm_vec_B = []

steps = int(T/dt)+1

# ODE solution
sol_x = np.zeros((steps+1,N))
#sol_x[0,:] = x0

# reference solution
ref_x = np.zeros((steps+1,N))
#ref_x[0,:] = x0

# set input
u = np.zeros([steps+1,M])
u[:,0] = 10.*np.sin(2.*np.pi*np.linspace(0,T,steps+1))
u[:,1] =  5.*np.cos(4.*np.pi*np.linspace(0,T,steps+1))

plt.figure(4)
time_vec = np.linspace(0,T,steps+1)
plt.plot(time_vec,u[:,0])
plt.plot(time_vec,u[:,1])
plt.xlabel('t')
plt.ylabel(r'$u(t)$')
plt.savefig('input_det.png')
#plt.show()

def init_B():

    B = 1./N*np.random.rand(N,M)

    return B

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

    A = -0.1 + 0.2*np.random.rand(N,N)

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

def solve_forward(x0,J,R,Q,B):

    xn = np.zeros(sol_x.shape)
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

def solve_backward(J,R,Q,B,y,ydata):

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

    #print(test_symm(sym))

    return sym

def make_skew(S):

    skew = 0.5*(S - S.T)

    #print(test_skew(skew))

    return skew


def compute_gradient_B(x,y,Q,B,u,ydata):

    #print('Muss noch angepasst werden!')

    G2 = np.zeros(B.shape)
    for i in range(steps):
        G2 = G2 + dt*np.outer(np.dot(Q,x[i,:]),y[i,:]-ydata[i,:]) #+ dt*np.outer(np.dot(Q.transpose(),np.dot(B,y[i,:]-ydata[i,:])),i*dt*u[i,:])

    return G2

def compute_gradient_Q(x,y,p,Q,B,u,ydata):

    IR = spl.inv(Q)
    G1 = lam*IR.dot(spl.logm(Q))
    G2 = np.zeros(Q.shape)
    for i in range(steps):
        # first term from dynamic # second term from cost functional which only depends on y = B^T Q z not active and not inside dynamic, should be fine
        # old version G2 = G2 + dt*( i*dt*np.outer(x[0,:],(-Q.T).dot(p[i,:])) - np.outer(x[0,:],p[i,:]) )#+ np.outer(x[i,:], np.dot(B, y[i,:] - ydata[i,:]))  )
        G2 = G2 + dt * np.outer(x[i, :], p[i, :])
    return make_symmetric(G1 + G2)

def compute_gradient_J(x,p,J):

    #print('Muss noch angepasst werden!')

    G2 = np.zeros(J.shape)
    for i in range(steps):
        #G2 = G2 + dt*( -i*dt*np.outer(x[0,:],p[i,:]) - np.outer(x[0,:],p[i,:]) )
        G2 = G2 + dt * np.outer(p[i, :],x[i, :])
    return make_skew(G2)

def compute_gradient_R(x,p,R):

    #print('Muss noch angepasst werden!')

    G2 = np.zeros(R.shape)
    for i in range(steps):
        #G2 = G2 + dt*( i*dt*np.outer(x[0,:],np.dot(R,R.dot(p[i,:]))) + dt*(i*dt*np.outer(R.dot(x[0,:]),R.dot(p[i,:]))) + np.outer(x[0,:],R.dot(p[i,:])) + np.outer(R.dot(x[0,:]),p[i,:]) )
        G2 = G2 + dt* ( np.outer(x[i,:], np.dot(R,p[i,:])) + np.outer(p[i,:], np.dot(R,x[i,:])) )
    return make_symmetric(G2)

def compute_gradient_x(p):

    return p[0,:]

def compute_retraction_x(x0,xi0):

    return x0 - xi0

def compute_retraction_B(R,XI):

    return R-XI

def compute_retraction_Q(Q,XI):

    SRR = spl.sqrtm(Q)
    ISRR = spl.inv(SRR)
    EXPt = spl.expm(ISRR.dot(XI.dot(ISRR)))
    G = SRR.dot(EXPt.dot(SRR))

    # alternative
    if alternative:
        IR = spl.inv(Q)
        G = R.dot(spl.expm(IR.dot(XI)))

    return G

def compute_retraction_J(J,XI):

    G = J-XI

    return G

def compute_retraction_R(R,XI):

    return R-XI

def evaluate_cost(y,ydata):

    return 0.5*np.sum(np.power(y-ydata,2))*dt

def armijo_rule(sol_x,sol_y,J,R,Q,B,XI_J,XI_R,XI_Q,XI_B,XI_x,ydata):
    sig = 1.
    fail_flag = False

    # updates
    #
    Jn  = compute_retraction_J(J,sig*XI_J)
    Rn  = compute_retraction_R(R,sig*XI_R)
    Qn  = compute_retraction_Q(Q,sig*XI_Q)
    Bn  = compute_retraction_B(B,sig*XI_B)
    x0n = x0 #compute_retraction_x(x0,sig*XI_x)

    c0 = evaluate_cost(sol_y, ydata)

    xn, yn = solve_forward(x0n,Jn,Rn,Qn,Bn)

    cn = evaluate_cost(yn, ydata)


    scalar_J = -np.sum(np.power(J, 2))
    scalar_R = -np.sum(np.power(R, 2))
    # Q pos def
    u, s, vh = np.linalg.svd(spl.logm(Q), full_matrices=True)
    scalar_Q = -s[0]
    scalar_B = -np.sum(np.power(B, 2))

    scalar = max(scalar_J,scalar_R,scalar_Q,scalar_B)
    #scalar = -0.001

    #print('cost diff', cn - c0)
    #print('gamma',gamma*sig*scalar)

    eps = 0.00000001

    while (cn-c0 > gamma*sig*scalar and sig > eps):
        sig = 0.5*sig

        #print('sig',sig)

        Jn  = compute_retraction_J(J,  sig * XI_J)
        Rn  = compute_retraction_R(R,  sig * XI_R)
        Qn  = compute_retraction_Q(Q,  sig * XI_Q)
        Bn  = compute_retraction_B(B,  sig * XI_B)
        x0n = x0 #compute_retraction_x(x0, sig * XI_x)

        xn, yn = solve_forward(x0n, Jn, Rn, Qn, Bn)

        cn = evaluate_cost(yn, ydata)
        #print('cost diff', cn - c0)

    if sig < eps:
        fail_flag = True

    return Jn, Rn, Qn, Bn, x0n, xn, yn, fail_flag, cn

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def remove_1_percent(R,neg_eps = 0.0000001):

    w, v = np.linalg.eig(R)

    #print('w',w)

    # remove negative
    w = w * (w >= neg_eps)
    # get sorting indices
    sumW = np.sum(w)
    ind = np.argsort(w)
    help_w = w[ind]

    l = w.shape[0]

    for ele in range(1,l):
        sumR = np.sum(help_w[:ele])
        if sumR/sumW > 0.01:
            break

    help_w[:ele] = 0
    w[ind] = help_w

    #print('helpw',help_w)

    return np.dot(np.diag(w), v)

def remove_k_smallest_eig(R,rem_k,neg_eps = 0.0000001):

    w, v = np.linalg.eig(R)

    #print('w', w)

    # remove negative
    w = w*(w>=neg_eps)
    # get sorting indices
    ind = np.argsort(w)
    help_w = w[ind]
    help_w[:rem_k] = 0

    #print('helpw', help_w)

    w[ind] = help_w

    return np.dot(np.diag(w),v)

# set true values to be identified later
BT = 10*init_B() #np.dot(A, A.transpose())
QT = 1./N*np.diag(range(1,N+1))
JT = generate_skew_symm(N)
RT = 1./N*np.diag(range(1,N+1)) #np.dot(A, A.transpose())
RT = remove_1_percent(RT)
xT = np.zeros([N,])  #0.1*np.ones(N,)

print('-----------------------------')
print('----- reference values ------')
print('BT',BT)
print('QT',QT)
print('JT',JT)
print('RT',RT)
print('xT',xT)
print('-----------------------------')

ref_x, ref_y = solve_forward(xT,JT,RT,QT,BT)
# test with noisy data
helpA,helpB = ref_y.shape
if noise_flag:
    ydata = ref_y + 0.005*np.random.randn(helpA,helpB)
else:
    ydata = copy.deepcopy(ref_y)
c0 = evaluate_cost(ref_y,ydata)

#print('ref_x',ref_x)

B = np.zeros([N,M])
B[:M,:M] = np.eye(M)
Q = np.eye(N)
J = generate_skew_symm(N)
R = np.zeros([N,N])
R[:int(N/10),:int(N/10)] = np.eye(int(N/10))
x0 = np.zeros([N,])  #1./N*np.arange(N) #np.zeros([N,]) # np.ones([N,])

norm_vec_Q.append(np.sqrt(np.sum(np.power(QT - Q, 2))))
norm_vec_J.append(np.sqrt(np.sum(np.power(JT - J, 2))))
norm_vec_R.append(np.sqrt(np.sum(np.power(RT - R, 2))))
norm_vec_B.append(np.sqrt(np.sum(np.power(BT - B, 2))))

print('norm B',np.sum(np.power(BT-B,2)))
print('norm Q',np.sum(np.power(QT-Q,2)))
print('norm J',np.sum(np.power(JT-J,2)))
print('norm R',np.sum(np.power(RT-R,2)))
print('norm x',np.sum(np.power(xT-x0,2)))

xn,yn = solve_forward(x0,J,R,Q,B)

plt.figure(7)
plt.plot(time_vec,yn)
plt.xlabel('t')
plt.ylabel(r'$y_\mathrm{initial}(t)$')
plt.savefig('initialoutput_det.png')

costStart = evaluate_cost(yn,ydata)
pvec = ref_y[::100]

cn = costStart # just a dummy value to start loop


#plt.plot(pvec,'o')
#plt.show()
cost_last = copy.deepcopy(costStart)

cost_vec.append(costStart)

print('y diff beginning', np.sum(abs(yn-ydata)*dt))

i=0
#while np.sum(abs(xn-ref_x)*dt) > dt and i < 150:
while (np.max(np.abs(yn-ydata)) > 0.001) and i<maxGradSteps:

    pn = solve_backward(J,R,Q,B,yn,ydata)

    # compute components of gradient
    XI_B = compute_gradient_B(xn,yn,Q,B,u,ydata)
    XI_Q = compute_gradient_Q(xn,yn,pn,Q,B,u,ydata)
    XI_J = compute_gradient_J(xn,pn,J)
    XI_R = compute_gradient_R(xn,pn,R)
    XI_x = compute_gradient_x(pn)


    if ~test_symm(XI_Q):
       #print('make symmetric')
       XI_Q = make_symmetric(XI_Q)

    if ~test_symm(XI_R):
       #print('make symmetric')
       XI_R = make_symmetric(XI_R)

    if ~test_skew(XI_J):
       XI_J = make_symmetric(XI_J)


    JA, RA, QA, BA, x0A, xA, yA, fail,cA = armijo_rule(xn,yn,J,R,Q,B,XI_J,XI_R,XI_Q,XI_B,XI_x,ydata)

    if  ~is_pos_def(QA):
        print('Q pos def', is_pos_def(QA))
        break


    if fail:
        print('Armijo failed')
        break
    else:
        J = JA
        R = RA
        Q = QA
        B = BA
        #x0 = x0A
        xn = xA
        yn = yA
        cost_last = copy.deepcopy(cn)
        cn = cA
        cost_vec.append(cn)
        norm_vec_Q.append(np.sqrt(np.sum(np.power(QT - Q, 2))))
        norm_vec_J.append(np.sqrt(np.sum(np.power(JT - J, 2))))
        norm_vec_R.append(np.sqrt(np.sum(np.power(RT - R, 2))))
        norm_vec_B.append(np.sqrt(np.sum(np.power(BT - B, 2))))


    print('y diff', np.sum(abs(yn-ydata)*dt))
    i += 1


if alternative:
    np.save('cost_alt.npy',cost_vec)
else:
    np.save('cost.npy', cost_vec)


print('---------End of simulation ------------')
print('# opt iterations:', i-1)

# print('J_ref',JT)
# print('J_opt',J)
#print('norm J',np.sum(np.abs(J-JT)))

# print('R_ref',RT)
# print('R_opt',R)
#print('norm R',np.sum(np.abs(R-RT)))

# print('Q_ref',QT)
# print('Q_opt',Q)
#print('norm Q',np.sum(np.abs(Q-QT)))

# print('B_ref',BT)
# print('B_opt',B)
#print('norm B',np.sum(np.abs(B-BT)))

# print('xT',xT)
# print('x0_opt',x0)

cFirst =cost_vec[0]
cLast = cost_vec[-1]
print('First cost', cFirst)
print('Last cost', cLast)
print('last cost is ', cLast/cFirst*100, '% of first cost')

plt.figure(1)
# plt.semilogy(cost_vec)
plt.plot(cost_vec)
plt.xlabel('iteration')
plt.ylabel(r'$\hat\mathcal J$')
plt.savefig('cost_det.png')

plt.figure(2)
plt.plot(time_vec,yn)
plt.xlabel('t')
plt.ylabel(r'$y(t)$')
plt.savefig('modeloutput_det.png')

plt.figure(3)
plt.plot(time_vec, ydata)
plt.xlabel('t')
plt.ylabel(r'$y_\mathrm{data}(t)$')
plt.savefig('dataoutput_det.png')

plt.figure(5)
plt.plot(time_vec, yn-ydata)
plt.xlabel('t')
plt.ylabel(r'$y(t) -y_\mathrm{data}(t)$')
plt.savefig('diffoutput_det.png')

plt.figure(6)
plt.plot(norm_vec_B)
plt.plot(norm_vec_Q)
plt.plot(norm_vec_J)
plt.plot(norm_vec_R)
plt.xlabel('t')
#plt.ylabel(r'$y(t) -y_\mathrm{data}(t)$')
plt.legend([r'$\|B-B_\mathrm{ref}\|_F$',r'$\|Q-Q_\mathrm{ref}\|_F$', r'$\|J-J_\mathrm{ref}\|_F$', r'$\|R-R_\mathrm{ref}\|_F$'])
plt.savefig('diffnorms_det.png')


plt.show()