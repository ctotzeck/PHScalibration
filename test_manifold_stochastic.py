''' Code written by C. Totzeck, Dec 2023

    Run the code to obtain the data and results of the first test case presented in the article
    'Data-driven adjoint-based calibration of port-Hamiltonian systems in time domain'
    by Michael Günther, Birgit Jacob and Claudia Totzeck (IMACM, University of Wuppertal)

'''
from helper_functions import *

noise_vals = [0.01, 0.05, 0.25]

for noi in noise_vals:

    noise_flag, M, N, maxGradSteps, T, steps, dt, lam = initialilze_problem_parameters_sto(True) # True for noise
    alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec = initialize_solver_parameters_sto(steps,N,T)
    cost_vec, norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B = initialize_container_postprocessing()

    u = initialize_input_sto(M,T,t_vec,steps)
    if noi == 0.01:
        BT, QT, JT, RT, xT = initialize_reference_data_sto(N,M)

    print_reference(BT,QT,JT,RT,xT)

    ref_x, ref_y = solve_forward(xT,JT,RT,QT,BT,u,steps,N,dt)

    # add artificial noise?
    if noise_flag:
        helpA, helpB = ref_y.shape
        ydata = ref_y + noi*np.random.randn(helpA,helpB)
    else:
        ydata = copy.deepcopy(ref_y)

    # evaluate cost for reference output
    c0 = evaluate_cost(ref_y,ydata,lam,QT,dt)


    B, Q, J, R, x0 = initialize_for_optimization_sto(N,M)

    norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B = compute_norms(QT,Q,JT,J,RT,R,BT,B,norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B)

    xn,yn = solve_forward(x0,J,R,Q,B,u,steps,N,dt)

    plot_solution_initial_setting_sto(t_vec,yn)

    costStart = evaluate_cost(yn,ydata,lam,Q,dt)
    pvec = ref_y[::100]

    cn = costStart # just a dummy value to start loop
    cost_last = copy.deepcopy(costStart) # save for armijo-rule
    cost_vec.append(costStart) # update cost vector


    print('y diff max beginning', np.max(abs(yn-ref_y)))

    # optimization loop
    i=0
    while (np.max(np.abs(yn-ydata)) > 0.0001) and i<maxGradSteps:

        pn = solve_backward(J,R,Q,B,yn,ydata,sol_x,T,dt,steps)
        Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x = compute_gradient(xn,yn,pn,Q,B,J,R,u,ydata,steps,dt,lam)
        n1 = compute_norm(Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x)
        norm_old = copy.deepcopy(n1)

        if i == 0:
            # first step is gradient step
            dir_J, dir_R, dir_Q, dir_B, dir_x = -Grad_J, -Grad_R, -Grad_Q, -Grad_B, -Grad_x
        else:
            # then CG-steps
            dir_J, dir_R, dir_Q, dir_B, dir_x = CG_direction(Grad_J, Grad_R, Grad_Q, Grad_B, Grad_x, XI_B, XI_Q, XI_J,
                                                             XI_R, XI_x, n1, norm_old, i, N)

        dir_Q, dir_R, dir_J = check_matrix_properties(dir_Q, dir_R, dir_J)
        JA, RA, QA, BA, x0A, xA, yA, fail, cA = armijo_rule_sto(xn, yn, J, R, Q, B, x0, dir_J, dir_R, dir_Q, dir_B, dir_x, ydata, u, sig, gamma, fail_flag, eps, alternative, dt, steps, N, lam)

        if fail:
            print('armijo failed')
            break

        XI_J, XI_R, XI_Q, XI_B, XI_x = transport(J, JA, dir_J, R, RA, dir_R, Q, QA, dir_Q, B, BA, dir_B, x0, x0A, dir_x)
        J, R, Q, B, x0, xn, yn, cost_last, cn = update_if_successful(JA, RA, QA, BA, x0A, xA, yA, cn, cA, J, R, Q, B, x0, xn, yn, cost_last, fail)
        cost_vec.append(cn)
        norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B = compute_norms(QT, Q, JT, J, RT, R, BT, B, norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B)
        #print('y diff max', np.max(abs(yn-ydata)))
        i += 1

    if alternative:
        np.save('cost_alt.npy',cost_vec)
    else:
        np.save('cost{0}.npy'.format(noi), cost_vec)


    print('---------End of simulation ------------')
    print('y diff max end', np.max(abs(yn - ref_y)))
    print('# opt iterations:', i-1)
    plots_postprocessing_sto(cost_vec,t_vec,yn,ydata,norm_vec_B,norm_vec_Q,norm_vec_J,norm_vec_R,i)
    print('----- robustness test -- noi = {0} ------'.format(noi))
    u = input_robustness_test_sto(t_vec,T,steps,M)
    Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y = run_robustness_test_sto(xT,JT,RT,QT,BT,u,x0,J,R,Q,B,steps,N,dt)
    np.save('ytest{0}.npy'.format(noi), Rtest_y)
    np.save('ytestRef{0}.npy'.format(noi), Rtest_ref_y)
    plot_robustness_test_sto(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt)
# show plots

plt.close('all')

c1 = np.load('cost0.01.npy')
c2 = np.load('cost0.05.npy')
c3 = np.load('cost0.25.npy')


plt.figure(1)
plt.semilogy(c1, label = '0.01')
plt.semilogy(c2, label = '0.05')
plt.semilogy(c3, label = '0.25')
plt.xlabel('iteration', fontsize=13)
plt.ylabel(r'$\hat\mathcal J$', fontsize = 13)
plt.legend()
plt.tight_layout()
plt.savefig('noise_cost.png', dpi=300)

y1 = np.load('ytest0.01.npy')
y2 = np.load('ytest0.05.npy')
y3 = np.load('ytest0.25.npy')
yR = np.load('ytestRef0.25.npy')

plt.figure(2)
plt.plot(t_vec,y1[:,0], label = '0.01')
plt.plot(t_vec,y2[:,0], label = '0.05')
plt.plot(t_vec,y3[:,0], label = '0.25')
plt.plot(t_vec,yR[:,0], '--', label = 'ref')
plt.xlabel(r'$t$', fontsize=13)
plt.ylabel('output, first component', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('noise_first.png', dpi=300)


plt.figure(3)
plt.plot(t_vec,y1[:,1], label = '0.01')
plt.plot(t_vec,y2[:,1], label = '0.05')
plt.plot(t_vec,y3[:,1], label = '0.25')
plt.plot(t_vec,yR[:,1], '--', label = 'ref')
plt.xlabel(r'$t$', fontsize=13)
plt.ylabel('output, second component', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('noise_second.png', dpi=300)


plt.show()
