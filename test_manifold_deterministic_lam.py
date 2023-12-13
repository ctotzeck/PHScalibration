''' Code written by C. Totzeck, Dec 2022

    Run the code to obtain the data and results of the first test case presented in the article
    'Data-driven adjoint-based calibration of port-Hamiltonian systems in time domain'
    by Michael Günther, Birgit Jacob and Claudia Totzeck (IMACM, University of Wuppertal)

'''


from helper_functions import *
#

noise_flag, M, N, maxGradSteps, T, steps, dt, lam = initialilze_problem_parameters_lam()
alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec = initialize_solver_parameters_lam(steps,N,T)
cost_vec, norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B = initialize_container_postprocessing()

u = initialize_input_lam(M,T,t_vec,steps)
BT, QT, JT, RT, xT = initialize_reference_data_lam(N,M)

print_reference(BT,QT,JT,RT,xT)

ref_x, ref_y = solve_forward(xT,JT,RT,QT,BT,u,steps,N,dt)

# add artificial noise?
if noise_flag:
    helpA, helpB = ref_y.shape
    ydata = ref_y + 0.005*np.random.randn(helpA,helpB)
else:
    ydata = copy.deepcopy(ref_y)

# evaluate cost for reference output
c0 = evaluate_cost(ref_y,ydata,lam,QT, dt)


B, Q, J, R, x0 = initialize_for_optimization_lam(N,M)

norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B = compute_norms(QT,Q,JT,J,RT,R,BT,B,norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B)

xn,yn = solve_forward(x0,J,R,Q,B,u,steps,N,dt)

u_rob = input_robustness_test_lam(t_vec,T,steps,M)
x_rob_init,yn_rob_init = solve_forward(x0,J,R,Q,B,u_rob,steps,N,dt)

plot_solution_initial_setting_lam(t_vec,yn)

costStart = evaluate_cost(yn,ydata,lam,Q, dt)
pvec = ref_y[::100]

cn = costStart # just a dummy value to start loop
cost_last = copy.deepcopy(costStart) # save for armijo-rule
cost_vec.append(costStart) # update cost vector


print('y diff max beginning', np.max(abs(yn-ydata)))

# optimization loop
i=0
while (cn > 0.0003*cost_vec[0]) and i<maxGradSteps:

    pn = solve_backward(J,R,Q,B,yn,ydata,sol_x,T,dt,steps)
    Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x = compute_gradient(xn,yn,pn,Q,B,J,R,u,ydata,steps,dt,lam)
    n1 = compute_norm(Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x)
    norm_old = copy.deepcopy(n1)

    if i==0:
        # first step is gradient step
        dir_J, dir_R, dir_Q, dir_B, dir_x = -Grad_J, -Grad_R, -Grad_Q, -Grad_B, -Grad_x
    else:
        # then CG-steps
        dir_J, dir_R, dir_Q, dir_B, dir_x = CG_direction(Grad_J,Grad_R,Grad_Q,Grad_B,Grad_x, XI_B, XI_Q, XI_J, XI_R, XI_x, n1, norm_old, i, N)

    dir_Q, dir_R, dir_J = check_matrix_properties(dir_Q, dir_R, dir_J)
    JA, RA, QA, BA, x0A, xA, yA, fail, cA = armijo_rule_lam(xn,yn,J,R,Q,B, x0, dir_J,dir_R, dir_Q, dir_B, dir_x,ydata,u,sig,gamma,fail_flag,eps,alternative,dt,steps,N,lam)

    if fail:
        print('armijo failed')
        break

    XI_J, XI_R, XI_Q, XI_B, XI_x = transport(J,JA,dir_J, R, RA, dir_R, Q, QA, dir_Q, B, BA, dir_B, x0, x0A, dir_x)
    J, R, Q, B, x0, xn, yn, cost_last, cn = update_if_successful(JA,RA,QA,BA,x0A,xA,yA,cn,cA,J, R, Q, B, x0, xn, yn, cost_last,fail)

    cost_vec.append(cn)
    norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B = compute_norms(QT, Q, JT, J, RT, R, BT, B, norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B)

    i += 1

if alternative:
    np.save('cost_alt_lam.npy',cost_vec)
else:
    np.save('cost_lam.npy', cost_vec)

print('---------End of simulation ------------')
print('# opt iterations:', i-1)
print('y diff max end', np.max(abs(yn-ydata)))
plots_postprocessing_lam(cost_vec,t_vec,yn,ydata,norm_vec_B,norm_vec_Q,norm_vec_J,norm_vec_R,i)
print('----- robustness test ------')
u = input_robustness_test_lam(t_vec,T,steps,M)
Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y = run_robustness_test_lam(xT,JT,RT,QT,BT,u,x0,J,R,Q,B,steps,N,dt)
plot_robustness_test_lam(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt)
# show plots
plt.show()