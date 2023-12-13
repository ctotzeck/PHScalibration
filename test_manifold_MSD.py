''' Code written by C. Totzeck, Dec 2023

    Run the code to obtain the data and results of the mass-spring-damper chain test case presented in the article
    'Data-driven adjoint-based calibration of port-Hamiltonian systems in time domain'
    by Michael Günther, Birgit Jacob and Claudia Totzeck (IMACM, University of Wuppertal)

'''


from helper_functions import *

# difference model sizes to check model order reduction
N_vec = [20,40,60,80,100,120]

for currN in N_vec:


    BT, QT, JT, RT, xT = initialize_reference_data_MSD()
    [NT,MT] = BT.shape

    noise_flag, M, N, maxGradSteps, T, steps, dt, lam = initialilze_problem_parameters_MSD(MT, currN)
    alternative, gamma, sig, fail_flag, eps, sol_x, ref_x, t_vec = initialize_solver_parameters_MSD(steps,N,T)
    cost_vec, norm_vec_Q, norm_vec_J, norm_vec_R, norm_vec_B = initialize_container_postprocessing()

    u = initialize_input_MSD(M, T, t_vec, steps)

    ref_x, ref_y = solve_forward(xT, JT, RT, QT, BT, u, steps, NT, dt)

    #print_reference(BT,QT,JT,RT,xT)




    save_initial_ref_setting_MSD(t_vec, ref_y, currN)

    # add artificial noise?
    if noise_flag:
        helpA, helpB = ref_y.shape
        ydata = ref_y + 0.005*np.random.randn(helpA,helpB)
    else:
        ydata = copy.deepcopy(ref_y)

    # evaluate cost for reference output
    c0 = evaluate_cost(ref_y,ydata,lam,QT,dt)


    B, Q, J, R, x0 = initialize_for_optimization_MSD(N,M)



    xn,yn = solve_forward(x0,J,R,Q,B,u,steps,N,dt)

    plot_and_save_solution_initial_setting_MSD(t_vec,yn,currN)

    costStart = evaluate_cost(yn,ydata,lam,Q,dt)
    pvec = ref_y[::100]

    cn = costStart # just a dummy value to start loop
    cost_last = copy.deepcopy(costStart) # save for armijo-rule
    cost_vec.append(costStart) # update cost vector

    ydiffmax = np.max(abs(yn-ydata))
    print('y diff max beginning', ydiffmax)

    # optimization loop
    i=0
    while i<maxGradSteps:

        pn = solve_backward(J,R,Q,B,yn,ydata,sol_x,T,dt,steps)
        Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x = compute_gradient(xn,yn,pn,Q,B,J,R,u,ydata,steps,dt,lam)
        n1 = compute_norm(Grad_B, Grad_Q, Grad_J, Grad_R, Grad_x)
        norm_old = copy.deepcopy(n1)

        if i == 0:
            # first step is gradient step
            dir_J, dir_R, dir_Q, dir_B, dir_x = -Grad_J, -Grad_R, -Grad_Q, -Grad_B, -Grad_x
        else:
            # then CG-steps
            dir_J, dir_R, dir_Q, dir_B, dir_x = CG_direction(Grad_J, Grad_R, Grad_Q, Grad_B, Grad_x, XI_B, XI_Q, XI_J, XI_R, XI_x, n1, norm_old, i, N)

        dir_Q, dir_R, dir_J = check_matrix_properties(dir_Q, dir_R, dir_J)
        JA, RA, QA, BA, x0A, xA, yA, fail, cA = armijo_rule_MSD(xn, yn, J, R, Q, B, x0, dir_J, dir_R, dir_Q, dir_B, dir_x, ydata, u, sig, gamma, fail_flag, eps, alternative, dt, steps, N,lam)

        if fail:
            print('armijo failed')
            break

        XI_J, XI_R, XI_Q, XI_B, XI_x = transport(J, JA, dir_J, R, RA, dir_R, Q, QA, dir_Q, B, BA, dir_B, x0, x0A, dir_x)
        J, R, Q, B, x0, xn, yn, cost_last, cn = update_if_successful(JA, RA, QA, BA, x0A, xA, yA, cn, cA, J, R, Q, B, x0, xn, yn, cost_last, fail)
        cost_vec.append(cn)

        i += 1

    if alternative:
        np.save('cost_alt.npy',cost_vec)
    else:
        np.save('cost_MSD{0}.npy'.format(currN), cost_vec)

    print('---------End of simulation ------------')
    ydiffmax = np.max(abs(yn-ydata))
    print('y diff max end', ydiffmax)
    np.save('ydiffmax_MSD{0}'.format(currN), ydiffmax)
    np.save('yopt_MSD{0}'.format(currN), yn)
    print('# opt iterations:', i-1)
    plots_postprocessing_MSD(cost_vec,t_vec,yn,ydata,norm_vec_B,norm_vec_Q,norm_vec_J,norm_vec_R,i)
    print('----- robustness test ------')
    u = input_robustness_test_MSD(t_vec,T,steps,M)
    Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y = run_robustness_test_MSD(xT,JT,RT,QT,BT,u,x0,J,R,Q,B,steps,NT,N,dt)
    np.save('ytest_MSD{0}.npy'.format(currN), Rtest_y)
    np.save('ytestRef_MSD{0}.npy'.format(currN), Rtest_ref_y)
    plot_robustness_test_MSD(Rtest_ref_x, Rtest_ref_y, Rtest_x, Rtest_y, t_vec, dt)


plot_model_reduction_MSD(t_vec,N_vec)
