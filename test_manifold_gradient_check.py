''' Code written by C. Totzeck, Dec 2022

    Run the code to obtain the data and results of the first test case presented in the article
    'Data-driven adjoint-based calibration of port-Hamiltonian systems in time domain'
    by Michael Günther, Birgit Jacob and Claudia Totzeck (IMACM, University of Wuppertal)

'''
from helper_functions import *

noise_flag, M, N, maxGradSteps, T, steps, dt, lam = initialilze_problem_parameters()
alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec = initialize_solver_parameters(steps,N,T)

BT, QT, JT, RT, xT = initialize_reference_data(N,M)
u = initialize_input(M,T,t_vec,steps)

# compute reference solution for cost functional
ref_x, ref_y = solve_forward(xT,JT,RT,QT,BT,u,steps,N,dt)


B, Q, J, R, x0 = initialize_for_optimization(N,M)
HB, HQ, HJ, HR, HR, Hx = initialize_directions(R,N,M)

xn,yn = solve_forward(x0,J,R,Q,B,u,steps,N,dt)
c = evaluate_cost(ref_y,yn,lam,Q,dt)

step_size = 0.001

B1 = compute_retraction_B(B,step_size*HB)
Q1 = compute_retraction_Q(Q,step_size*HQ)
J1 = compute_retraction_J(J,step_size*HJ)
R1 = compute_retraction_R(R,step_size*HR)
x1 = compute_retraction_x(x0,step_size*Hx)


xn1B,yn1B = solve_forward(x0,J,R,Q,B1,u,steps,N,dt)
c1 = evaluate_cost(ref_y,yn1B,lam,Q,dt)

finite_diff_B= (c1-c)/step_size
#print('finite_difference value B', finite_diff_B)


xn1J,yn1J = solve_forward(x0,J1,R,Q,B,u,steps,N,dt)
c1 = evaluate_cost(ref_y,yn1J,lam,Q,dt)

finite_diff_J = (c1-c)/step_size
#print('finite_difference value J', finite_diff_J)

xn1Q,yn1Q = solve_forward(x0,J,R,Q1,B,u,steps,N,dt)
c1 = evaluate_cost(ref_y,yn1Q,lam,Q1,dt)

finite_diff_Q = (c1-c)/step_size
#print('finite_difference value Q', finite_diff_Q)

xn1R,yn1R = solve_forward(x0,J,R1,Q,B,u,steps,N,dt)
c1 = evaluate_cost(ref_y,yn1R,lam,Q,dt)

finite_diff_R = (c1-c)/step_size
#print('finite_difference value R', finite_diff_R)


xn1x,yn1x = solve_forward(x1,J,R,Q,B,u,steps,N,dt)
c1 = evaluate_cost(ref_y,yn1x,lam,Q,dt)

finite_diff_x = (c1-c)/step_size
#print('finite_difference value x', finite_diff_x)

xn1all,yn1all = solve_forward(x1,J1,R1,Q1,B1,u,steps,N,dt)
c1 = evaluate_cost(ref_y,yn1all,lam,Q1,dt)

finite_diff_all_at_once = (c1-c)/step_size



# compute gradient
pn = solve_backward(J,R,Q,B,yn,ref_y,sol_x,T,dt,steps)
Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x = compute_gradient(xn,yn,pn,Q,B,J,R,u,ref_y,steps,dt,lam)

inner_B = compute_inner_B(B,Grad_B,HB)
#print('inner_B', inner_B)
inner_J = compute_inner_J(J,Grad_J,HJ)
#print('inner_J', inner_J)
inner_Q = compute_inner_Q(Q,Grad_Q,HQ)
#print('inner_Q', inner_Q)

inner_R = compute_inner_R(R,Grad_R,HR)
#print('inner_R', inner_R)
inner_x = compute_inner_x(x0,Grad_x,Hx)
#print('inner_x', inner_x)

sum_inner = inner_B + inner_Q + inner_J + inner_R + inner_x
sum_fd = finite_diff_B + finite_diff_Q + finite_diff_J + finite_diff_R + finite_diff_x
print('inner product (grad*h)          :', sum_inner)
print('finite diffence (sum individual):', sum_fd)
print('finite difference (all at once) :', finite_diff_all_at_once)

print('inner product relative to all at one:', abs(sum_inner-finite_diff_all_at_once)/abs(finite_diff_all_at_once))
print('finite diffence relative to individual:', abs(sum_inner-sum_fd)/abs(sum_fd))


# print('###################')
relB = np.abs(inner_B - finite_diff_B)/np.abs(finite_diff_B)
relJ = np.abs(inner_J - finite_diff_J)/np.abs(finite_diff_J)
relQ = np.abs(inner_Q - finite_diff_Q)/np.abs(finite_diff_Q)
relR = np.abs(inner_R - finite_diff_R)/np.abs(finite_diff_R)
relX = np.abs(inner_x - finite_diff_x)/np.abs(finite_diff_x)
# print('diff B', relB, inner_B, finite_diff_B)
# print('diff J', relJ, inner_J, finite_diff_J)
# print('diff Q', relQ, inner_Q, finite_diff_Q)
# print('diff R', relR, inner_R, finite_diff_R)
# print('diff x', relX, inner_x, finite_diff_x)

if (relB < 0.01)*(relJ < 0.01)*(relQ < 0.01)*(relR < 0.01)*(relX < 0.01):
    print('Gradient check successful!    :-)')