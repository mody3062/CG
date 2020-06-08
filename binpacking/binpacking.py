import pandas as pd
import numpy as np
import cplex
from cplex.exceptions import CplexSolverError
from cplex import SparsePair
from cplex.six.moves import zip
import time
import numba
from ortools.algorithms import pywrapknapsack_solver
from numba import jit
import matplotlib.pyplot as plt
import csv
import pickle
import ctypes
import sys
from sklearn.cluster import KMeans

knapsack = ctypes.CDLL('knapsack.so')
knapsack.knapsack_bnb.restype = ctypes.c_double


class binpacking:
    def __init__(self, prob_num, type, sep):
        self.prob = prob_num

        self.sep = sep

        self.type = type

        self.data()

        if self.type[0] == 'u':
            self.W = 1500

        else :
            self.W = 1000

    def data(self):
        test = pd.read_csv("data/%s.txt"%(self.type), sep="\n", header=None)

        ind = list(test[test[0].str.contains(self.type[0])].index)

        mul = 10
        probs = [[int(float(test[0][i])*mul) for i in range(ind[j] + 2, ind[j + 1])] for j in range(len(ind) - 1)]
        probs.append([int(float(test[0][i])*mul) for i in range(ind[len(ind) - 1] + 2, len(test[0]))])

        self.w = probs[self.prob]
        self.I = range(len(self.w))

    @staticmethod
    def KnapsackBnB(profits, weights, b):

        n = len(profits)

        zero_list = [0] * n
        x = (ctypes.c_int * n)(*zero_list)
        one_list = [1] * n
        m = (ctypes.c_int * n)(*one_list)

        p = (ctypes.c_double * n)(*profits)
        w = (ctypes.c_int * n)(*weights)

        return knapsack.knapsack_bnb(n, p, w, m, x, b), list(x)

    def Knapsack2(val, wt, W):

        n = len(val)
        K = np.zeros((n + 1, W + 1))
        for i in range(n + 1):
            for w in range(W + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif wt[i - 1] <= w:
                    K[i][w] = max(val[i - 1]
                                  + K[i - 1][w - wt[i - 1]],
                                  K[i - 1][w])
                else:
                    K[i][w] = K[i - 1][w]

        res = K[n][W]
        sol = np.zeros(n)
        w = W
        print(W)
        for i in range(n, 0, -1):
            if res <= 0:
                break
            elif res == K[i - 1][w]:
                continue
            else:
                sol[i - 1] = 1
                res = res - val[i - 1]
                w = w - wt[i - 1]
        return K[n][W], sol



    def Kelly(self):

        w = self.w
        I = range(len(w))

        M = cplex.Cplex()

        var = list(range(len(w)))

        M.variables.add(obj=[1] * len(var), lb=[0] * len(var))

        M.linear_constraints.add(lin_expr=[SparsePair()] * len(w),
                                 senses=["G"] * len(w),
                                 rhs=[1] * len(w))
        for i in range(len(w)):
            M.linear_constraints.set_coefficients(i, i, 1)

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        start = time.time()
        mt = 0
        st = 0
        ite = 0
        solutions = []
        iterations = []
        criteria = True

        while criteria:

            ite += 1

            M.set_problem_type(M.problem_type.LP)
            ct = time.time()

            M.solve()
            solutions.append(float(M.solution.get_objective_value()))
            iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
            mt += time.time() - ct

            pi = list(M.solution.get_dual_values())[:len(w)]
            dual = list(M.solution.get_dual_values())

            v = pi
            W = self.W

            pt = time.time()
            # print(w)

            S_obj, sol = binpacking.KnapsackBnB(v, w, W)

            # print(S_obj, sol)

            st += time.time() - pt

            if S_obj - 0.000001 > 1.:

                criteria = True
                newsub = sol
                idx = M.variables.get_num()
                M.variables.add(obj=[1.0])
                M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                               [idx] * len(var),
                                                               newsub)))

                var.append(idx)
            else:
                criteria = False

        M.set_problem_type(M.problem_type.LP)
        ct = time.time()
        M.solve()
        # M.write('kelly.lp')
        solutions.append(float(M.solution.get_objective_value()))
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        mt += time.time() - ct
        tt = time.time() - start

        self.Kelly_M = M

        self.Kelly_result = [

            'Kelly', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]

    def Stabilization(self):

        eps = 0.1

        w = self.w
        I = range(len(w))

        M = cplex.Cplex()

        var = list(range(len(w)))
        vals = np.zeros((len(w), len(var)))

        np.fill_diagonal(vals, 1)

        x_p = lambda p: 'x_%d' % (p)

        x = [x_p(p) for p in range(len(var))]

        M.variables.add(
            lb=[0] * len(x),
            ub=[cplex.infinity] * len(x),
            names=x,
            obj=[1.] * len(x),
            types=['C'] * len(x)
        )

        dp_i = lambda i: 'dp_%d' % (i)

        dp = [dp_i(i) for i in I]

        M.variables.add(
            lb=[0] * len(dp),
            ub=[eps] * len(dp),
            names=dp,
            obj=[0] * len(dp),
            types=['C'] * len(dp)
        )



        dm_i = lambda i: 'dm_%d' % (i)

        dm = [dm_i(i) for i in I]

        M.variables.add(
            lb=[0] * len(dm),
            ub=[eps] * len(dm),
            names=dm,
            obj=[0] * len(dm),
            types=['C'] * len(dm)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [dm[i]] + [dp[i]],
                    val=list(vals[i]) +[-1.0] + [1.0]
                )
                for i in I
            ],
            senses=["G" for i in w],
            rhs=[1. for i in w])



        # M.linear_constraints.add(lin_expr=[SparsePair()] * len(w),
        #                          senses=["G"] * len(w),
        #                          rhs=[1] * len(w))
        # for i in range(len(w)):
        #     M.linear_constraints.set_coefficients(i, i, 1)

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        start = time.time()
        mt = 0
        st = 0
        ite = 0
        solutions = []
        iterations = []
        criteria = True

        while criteria:

            ite += 1

            M.set_problem_type(M.problem_type.LP)
            ct = time.time()

            M.solve()
            solutions.append(float(M.solution.get_objective_value()))
            iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
            mt += time.time() - ct

            pi = list(M.solution.get_dual_values())[:len(w)]
            dual = list(M.solution.get_dual_values())

            v = pi
            W = self.W

            pt = time.time()
            # print(w)

            S_obj, sol = binpacking.KnapsackBnB(v, w, W)

            # print(S_obj, sol)

            st += time.time() - pt

            if S_obj - 0.000001 > 1. or eps != 0 :

                criteria = True
                newsub = sol
                idx = M.variables.get_num()
                M.variables.add(obj=[1.0])
                M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                               [idx] * len(var),
                                                               newsub)))

                var.append(idx)

                if ite % 100 == 0:
                    eps *= 0.1
                    if ite == 600:
                        eps = 0

                    for dv in dm + dp:
                        M.variables.set_upper_bounds(dv, eps)

            else:
                criteria = False

        M.set_problem_type(M.problem_type.LP)
        ct = time.time()
        M.solve()
        # M.write('kelly.lp')
        solutions.append(float(M.solution.get_objective_value()))
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        mt += time.time() - ct
        tt = time.time() - start

        self.Stab_M = M

        self.Stab_Result = [

            'Stabilization', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]

    def Sep_Stab(self):

        eps = 0.1
        # sep = 2

        w = self.w

        I = range(len(w))
        Is = list(np.array_split(I, self.sep))
        BigM = 100000

        M = cplex.Cplex()

        var = list(range(len(w)))

        vals = np.zeros((len(w), len(var)))

        np.fill_diagonal(vals, 1)

        x_p = lambda p: 'x_%d' % (p)

        x = [x_p(p) for p in range(len(var))]

        M.variables.add(
            lb=[0] * len(x),
            ub=[cplex.infinity] * len(x),
            names=x,
            obj=[1.] * len(x),
            types=['C'] * len(x)
        )

        y_i = lambda i: 'y_%d' % (i)

        y = [y_i(i) for i in I]

        ys = [[y[i] for i in Is[j]] for j in range(self.sep)]

        dp_i = lambda i: 'dp_%d' % (i)

        dp = [dp_i(i) for i in I]

        M.variables.add(
            lb=[0] * len(dp),
            ub=[eps] * len(dp),
            names=dp,
            obj=[0] * len(dp),
            types=['C'] * len(dp)
        )

        dm_i = lambda i: 'dm_%d' % (i)

        dm = [dm_i(i) for i in I]

        M.variables.add(
            lb=[0] * len(dm),
            ub=[eps] * len(dm),
            names=dm,
            obj=[0] * len(dm),
            types=['C'] * len(dm)
        )

        M.variables.add(
            # lb=[0] * len(y),
            # ub=[cplex.infinity] * len(y),
            names=y,
            obj=[BigM] * len(y),
            types=['C'] * len(y)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [y[i]] + [dm[i]] + [dp[i]],
                    val=list(vals[i]) + [1.0]  + [-1.0] + [1.0]
                )
                for i in I
            ],
            senses=["G" for i in w],
            rhs=[1. for i in w])

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        start = time.time()
        mt = 0
        st = 0
        ite = 0
        s2 = 0
        solutions = []
        iterations = []
        penalty = 0.6

        for twice in range(2):

            for sec in range(self.sep):


                criteria = True
                y_fix = list(set(y) - set(list(ys[sec])))
                I_fix = list(set(I) - set(list(Is[sec])))
                # M.objective.set_linear(zip(y_fix, [BigM] * len(y_fix)))
                M.variables.set_upper_bounds(zip(y_fix, np.zeros(len(y_fix))))
                M.variables.set_lower_bounds(zip(y_fix, [0] * len(y_fix)))

                while criteria:
                    ite += 1

                    if ite % 500 == 0:
                        penalty = penalty * 0.6
                    if penalty < 0.1:
                        penalty = 100
                        # print(penalty)

                    M.set_problem_type(M.problem_type.LP)

                    ct = time.time()
                    M.solve()

                    solutions.append(float(M.solution.get_objective_value()))
                    iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

                    mt += time.time() - ct
                    dual = list(M.solution.get_dual_values())

                    pi = dual

                    pi_ = [dual[i] * penalty for i in I_fix]

                    v = pi
                    W = self.W

                    pt = time.time()
                    #####################################################
                    # if ite >= 51 and ite<=162:
                    #     S_obj, sol =binpacking.Knapsack2(v,w,W)
                    #     s2 += time.time() - pt

                    # # else:
                    #     aa = time.time()
                    #     S_obj, sol = binpacking.KnapsackBnB(v, w, W)
                    #     st += time.time() - pt
                    #####################################################

                    S_obj, sol = binpacking.KnapsackBnB(v, w, W)

                    st += time.time() - pt


                    if ite % 100 == 0:
                        eps *= 0.1
                        if ite == 600:
                            eps = 0

                        for dv in dm + dp:
                            M.variables.set_upper_bounds(dv, eps)

                    if S_obj - 0.00001 > 1.:

                        criteria = True
                        M.objective.set_linear(list(zip(list(map(lambda x: int(x + len(w)), Is[sec])), pi_)))

                        # M.objective.set_linear(zip(y_fix, [BigM] * len(y_fix)))
                        newsub = sol
                        idx = M.variables.get_num()
                        M.variables.add(obj=[1.0])
                        M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                                       [idx] * len(var),
                                                                       newsub)))

                        var.append(idx)
                        # if ite >= 50:
                        #     print('2')


                    else:
                        criteria = False
                        # M.write('1.lp')

                if M.solution.get_values(ys[sec]) == [0] * len(ys[sec]):
                    break
            else:
                continue
            break

        M.set_problem_type(M.problem_type.LP)

        ct = time.time()
        M.solve()
        # M.write('sep.lp')
        solutions.append(float(M.solution.get_objective_value()))
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        print(s2)
        mt += time.time() - ct
        tt = time.time() - start
        self.Sep_Stab_M = M

        self.Sep_Stab_Result = [

            'Seep_Stab', ite, mt, st, tt, mt / (st + mt), solutions,
            np.average(np.array(iterations))

        ]

    @staticmethod
    def cluster(clusters, a, c, J):

        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(c, a)

        labels = kmeans.labels_

        Js = [[] for _ in range(clusters)]

        for i in J:
            Js[labels[i]].append(i)

        return Js

    def Separation(self):
        # sep = 2

        w = self.w

        I = range(len(w))
        Is = list(np.array_split(I, self.sep))
        BigM = 100000

        M = cplex.Cplex()

        var = list(range(len(w)))

        vals = np.zeros((len(w), len(var)))

        np.fill_diagonal(vals, 1)

        x_p = lambda p: 'x_%d' % (p)

        x = [x_p(p) for p in range(len(var))]

        M.variables.add(
            lb=[0] * len(x),
            ub=[cplex.infinity] * len(x),
            names=x,
            obj=[1.] * len(x),
            types=['C'] * len(x)
        )

        y_i = lambda i: 'y_%d' % (i)

        y = [y_i(i) for i in I]

        ys = [[y[i] for i in Is[j]] for j in range(self.sep)]

        M.variables.add(
            # lb=[0] * len(y),
            # ub=[cplex.infinity] * len(y),
            names=y,
            obj=[BigM] * len(y),
            types=['C'] * len(y)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [y[i]],
                    val=list(vals[i]) + [1.0]
                )
                for i in I
            ],
            senses=["G" for i in w],
            rhs=[1. for i in w])

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)


        start = time.time()
        mt = 0
        st = 0
        ite = 0
        s2 =0
        solutions = []
        iterations = []
        penalty = 0.6

        for twice in range(2):

            for sec in range(self.sep):

                print(ite)

                criteria = True
                y_fix = list(set(y) - set(list(ys[sec])))
                I_fix = list(set(I) - set(list(Is[sec])))
                # M.objective.set_linear(zip(y_fix, [BigM] * len(y_fix)))
                M.variables.set_upper_bounds(zip(y_fix, np.zeros(len(y_fix))))
                M.variables.set_lower_bounds(zip(y_fix, [0] * len(y_fix)))



                while criteria:
                    ite += 1

                    if ite % 500 ==0:
                        penalty = penalty * 0.6
                    if penalty < 0.1:
                        penalty = 100
                        # print(penalty)


                    M.set_problem_type(M.problem_type.LP)

                    ct = time.time()
                    M.solve()

                    solutions.append(float(M.solution.get_objective_value()))
                    iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

                    mt += time.time() - ct
                    dual = list(M.solution.get_dual_values())

                    pi = dual

                    pi_ = [dual[i]*penalty for i in I_fix]


                    v = pi
                    W = self.W

                    pt = time.time()
                    #####################################################
                    # if ite >= 51 and ite<=162:
                    #     S_obj, sol =binpacking.Knapsack2(v,w,W)
                    #     s2 += time.time() - pt

                    # # else:
                    #     aa = time.time()
                    #     S_obj, sol = binpacking.KnapsackBnB(v, w, W)
                    #     st += time.time() - pt
                    #####################################################

                    S_obj, sol = binpacking.KnapsackBnB(v, w, W)
                    if ite == 1000:
                        print(v,w,W)
                        print(time.time() - pt)
                    st += time.time() - pt


                    if S_obj - 0.00001 > 1.:

                        criteria = True
                        M.objective.set_linear(list(zip(list(map(lambda x: int(x + len(w)), Is[sec])), pi_)))

                        # M.objective.set_linear(zip(y_fix, [BigM] * len(y_fix)))
                        newsub = sol
                        idx = M.variables.get_num()
                        M.variables.add(obj=[1.0])
                        M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                                       [idx] * len(var),
                                                                       newsub)))

                        var.append(idx)
                        # if ite >= 50:
                        #     print('2')


                    else:
                        criteria = False
                        # M.write('1.lp')

                if M.solution.get_values(ys[sec]) == [0]*len(ys[sec]):
                    print('dddd')
                    break
            else:
                continue
            break



        M.set_problem_type(M.problem_type.LP)

        ct = time.time()
        M.solve()
        # M.write('sep.lp')
        solutions.append(float(M.solution.get_objective_value()))
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        print(s2)
        mt += time.time() - ct
        tt = time.time() - start
        self.Separation_M = M

        self.Separation_result = [

            'Separation %s'%(self.sep), ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]

    def chevy(self):

        w = self.w

        ######### Master Problem ###########

        M = cplex.Cplex()

        # Parameters
        var = list(range(len(w)))
        alpha = 1.0
        init_pi = sum(w) / 150
        epsilon = 0.1

        # decision varialbes types=["C"]*len(var)
        M.variables.add(obj=[1] * len(var), names=['x_' + str(i) for i in var], lb=[0]*len(var))
        M.variables.add(obj=[-init_pi], names='z', lb = [0])
        M.variables.add(names=['y_' + str(i) for i in list(range(len(w)))], lb=[0]*len(w))

        # pattern constraints
        vals = np.zeros((len(w), len(var)))

        np.fill_diagonal(vals, 1)

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=['x_' + str(j) for j in var] + ['y_' + str(i)] + ['z'],
                    val=list(vals[i]) + [-1.0] + [-1.0]
                )
                for i in range(len(w))
            ],
            senses=["G" for i in w],
            rhs=[0 for i in w])

        # chebyshev constraint
        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=['x_' + str(j) for j in var] + ['y_' + str(i) for i in range(len(w))] + ['z'],
                    val=[1.0 for k in var] + [1.0 for l in w] + [alpha * len(w) ** (1 / 2)]
                )
            ],
            senses=["G"],
            rhs=[1.0])


        M.objective.set_sense(M.objective.sense.minimize)


        M.write('cheby.lp')


        ite = 0
        while True:
            ite += 1
            #         M.write('cheby_m.lp')

            M.set_log_stream(None)
            M.set_error_stream(None)
            M.set_warning_stream(None)
            M.set_results_stream(None)



            # M.write('cheby.lp')

            self.chevy_M = M
            M.solve()

            v = [pie for pie in M.solution.get_dual_values(list(range(len(w))))]
            # S.objective.set_linear(list(zip(list(range(len(w))), price)))
            # #         S.write('cheby_s.lp')
            # S.set_log_stream(None)
            # S.set_error_stream(None)
            # S.set_warning_stream(None)
            # S.set_results_stream(None)
            # S.solve()

            S_obj, sol = binpacking.KnapsackBnB(v, w, W)
            print(sol)



            if M.solution.get_objective_value() < epsilon * M.solution.get_values('z'):
                break

            if S_obj < 1 + 1.0e-6:
                newsub = sol
                idx = M.variables.get_num()
                M.variables.add(obj=[1.0])
                M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                               [idx] * len(var),
                                                               newsub)))
                var.append(idx)


            else:
                new_pi = M.solution.get_dual_values()
                M.objective.set_linear('z', -sum(new_pi))

        M.variables.set_types(
            list(zip(var, [M.variables.type.continuous] * len(var))))
        M.solve()

        self.chevy_M = M

        self.chevy_result = [

            'Separation %s'%(self.sep), ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]



################# For Parallel ###################
    @staticmethod
    def Master_Kelly(w):


        # w = self.w
        I = range(len(w))

        M = cplex.Cplex()

        var = list(range(len(w)))

        M.variables.add(obj=[1] * len(var), lb=[0] * len(var))

        M.linear_constraints.add(lin_expr=[SparsePair()] * len(w),
                                 senses=["G"] * len(w),
                                 rhs=[1] * len(w))
        for i in range(len(w)):
            M.linear_constraints.set_coefficients(i, i, 1)

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        return M, var

    @staticmethod
    def Master_Sep(w, sep):


        I = range(len(w))
        Is = list(np.array_split(I, sep))
        BigM = 100000

        M = cplex.Cplex()

        var = list(range(len(w)))

        vals = np.zeros((len(w), len(var)))

        np.fill_diagonal(vals, 1)

        x_p = lambda p: 'x_%d' % (p)

        x = [x_p(p) for p in range(len(var))]

        M.variables.add(
            lb=[0] * len(x),
            ub=[cplex.infinity] * len(x),
            names=x,
            obj=[1.] * len(x),
            types=['C'] * len(x)
        )

        y_i = lambda i: 'y_%d' % (i)

        y = [y_i(i) for i in I]

        ys = [[y[i] for i in Is[j]] for j in range(sep)]

        M.variables.add(
            # lb=[0] * len(y),
            # ub=[cplex.infinity] * len(y),
            names=y,
            obj=[BigM] * len(y),
            types=['C'] * len(y)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [y[i]],
                    val=list(vals[i]) + [1.0]
                )
                for i in I
            ],
            senses=["G" for i in w],
            rhs=[1. for i in w])

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        return M, var, y, ys


    def Kelly_CG(self, M, tolerence, var):

        w = self.w


        mt = 0
        st = 0
        ite = 0
        solutions = []
        iterations = []
        criteria = True


        while criteria:

            ite += 1

            M.set_problem_type(M.problem_type.LP)
            ct = time.time()

            M.solve()
            solutions.append(float(M.solution.get_objective_value()))
            iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
            mt += time.time() - ct

            pi = list(M.solution.get_dual_values())[:len(w)]
            dual = list(M.solution.get_dual_values())

            v = pi
            W = self.W

            pt = time.time()
            # print(w)

            S_obj, sol = binpacking.KnapsackBnB(v, w, W)

            # print(S_obj, sol)

            st += time.time() - pt

            if S_obj - tolerence > 1.:

                criteria = True
                newsub = sol
                idx = M.variables.get_num()
                M.variables.add(obj=[1.0])
                M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                               [idx] * len(var),
                                                               newsub)))

                var.append(idx)
            else:
                criteria = False

        M.set_problem_type(M.problem_type.LP)
        ct = time.time()
        M.solve()
        # M.write('kelly.lp')
        solutions.append(float(M.solution.get_objective_value()))
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        mt += time.time() - ct
        # tt = time.time() - start

        self.Kelly_M = M


        Kelly_result = [

            ite, mt, st,  solutions, iterations

        ]

        return Kelly_result


def save_results(K, S, sep, clustering):
    if clustering == True:

        with open('Results/result_%s_cluster.txt' % (sep), 'wb') as f:
            pickle.dump(K, f)
            pickle.dump(S, f)


    else:
        with open('Results/result_%s.txt' % (sep), 'wb') as f:
            pickle.dump(K, f)
            pickle.dump(S, f)


def result_table(K, S):
    name = ['method', 'iteration', 'M', 'S', 'total', 'M_per', 'sol', 'pivot_iteration']

    if K != {}:
        k = pd.DataFrame(K)
        k = k.transpose()
        k.columns = name
        k.drop(['M', 'S', 'sol'], axis=1, inplace=True)

    s = pd.DataFrame(S)
    s = s.transpose()
    s.columns = name
    s.drop(['M', 'S', 'sol'], axis=1, inplace=True)

    return pd.concat([k, s], axis=1)


def result_table2(K, S):
    results = [pd.DataFrame(S), pd.DataFrame(K)]
    table = pd.concat(results, axis=1)
    name = ['method', 'iteration', 'M', 'S', 'total', 'M_per', 'sol', 'pivot_iteration']

    table = table.transpose()

    table.columns = name

    table.drop(['M', 'S', 'sol'], axis=1, inplace=True)

    return table


def obj_change_graph(problems, K, S, sep):
    for i in range(len(problems) - 1):

        kelly = list(K[problems[i]][6])
        separation = list(S[problems[i]][6])

        if len(kelly) > len(separation):
            extra = [None] * (len(kelly) - len(separation))
            separation.extend(extra)

        else:
            extra = [None] * (len(separation) - len(kelly))
            kelly.extend(extra)

        x = range(len(separation))

        plt.figure(1, figsize=(8, 8))

        plt.legend(['kelly', 'separation'], loc='best', bbox_to_anchor=(0, -0.2, 1, 1), fontsize=5.5)

        pos = int('33%s' % (i + 1))

        plt.subplot(pos)

        plt.title(problems[i], x=0.8, y=0.8, fontsize=8)

        plt.plot(list(range(len(kelly))), kelly)
        plt.plot(list(range(len(kelly))), separation)

        plt.xticks(fontsize=5, rotation=0)
        plt.yticks(fontsize=5, rotation=30)

    words = 'Changes of objective values during iterations (sep : %s) \n x_axis: # of iterations, y_axis: objective value' % (
        sep)
    suptitle = plt.suptitle(words, x=0.45, y=0.87, fontsize=12)

    plt.tight_layout()
    plt.legend(['kelly', 'separation'], loc='best', bbox_to_anchor=(0, -0.2, 1, 1), fontsize=5.5)

    plt.subplots_adjust(top=0.8, right=0.8)

    plt.savefig('Results/obj_changes_%s.pdf' % (sep))

    plt.show()


def iteration_graph(problems, K, S, sep):
    kelly = []
    separation = []

    for problem in problems:
        kelly.append(float(K[problem][1]))
        separation.append(float(S[problem][1]))

    types = ['Kelly', 'Stab.']
    pos = np.arange(5)
    bar_width = 0.15
    plt.figure(2, figsize=(10, 8))

    plt.subplot(211)
    plt.bar(pos, kelly[:5], bar_width)
    plt.bar(pos + bar_width, separation[:5], bar_width)
    plt.xticks(pos, problems[:5], fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('# of iterations', fontsize=10)
    plt.legend(types, loc='best')

    plt.subplot(212)
    plt.bar(pos, kelly[5:], bar_width)
    plt.bar(pos + bar_width, separation[5:], bar_width)
    plt.xticks(pos, problems[5:], fontsize=8)
    plt.yticks(fontsize=8)

    plt.xlabel('Problems', fontsize=10)
    plt.ylabel('# of iterations', fontsize=10)
    plt.legend(types, loc='best')

    words = 'Performance comparison of the algorithms (sep : %s)' % (sep)
    suptitle = plt.suptitle(words, x=0.45, y=0.87, fontsize=14)
    plt.subplots_adjust(top=0.8, right=0.8)

    plt.savefig('Results/result_iterations_%s.pdf' % (sep))

    plt.show()


def iteration_graph2(problems, K, S, Sc, sep, pivot):
    if pivot == True:
        p = 7

    else:
        p = 1

    kelly = []
    separation = []
    sep_clu = []

    for problem in problems:
        kelly.append(float(K[problem][p]))
        separation.append(float(S[problem][p]))
        sep_clu.append(float(Sc[problem][p]))

    types = ['Kelly', 'Separation.', 'sep_cluster']
    pos = np.arange(5)
    bar_width = 0.15
    plt.figure(2, figsize=(10, 8))

    plt.subplot(211)
    plt.bar(pos, kelly[:5], bar_width)
    plt.bar(pos + bar_width, separation[:5], bar_width)
    plt.bar(pos + bar_width + bar_width, sep_clu[:5], bar_width)
    plt.xticks(pos + bar_width, problems[:5], fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('# of iterations', fontsize=10)
    plt.legend(types, loc='best')

    plt.subplot(212)
    plt.bar(pos, kelly[5:], bar_width)
    plt.bar(pos + bar_width, separation[5:], bar_width)
    plt.bar(pos + bar_width + bar_width, sep_clu[5:], bar_width)
    plt.xticks(pos + bar_width, problems[5:], fontsize=8)
    plt.yticks(fontsize=8)

    plt.xlabel('Problems', fontsize=10)
    plt.ylabel('# of iterations', fontsize=10)
    plt.legend(types, loc='best')

    words = 'Performance comparison of the algorithms (sep : %s)' % (sep)
    suptitle = plt.suptitle(words, x=0.45, y=0.87, fontsize=14)
    plt.subplots_adjust(top=0.8, right=0.8)

    plt.savefig('Results/result_iterations_%s.pdf' % (sep))

    plt.show()


def read_results(sep, clustering):
    if clustering == True:
        with open('Results/result_%s_cluster.txt' % (sep), 'rb') as f:
            K = pickle.load(f)
            S = pickle.load(f)

    else:
        with open('Results/result_%s.txt' % (sep), 'rb') as f:
            K = pickle.load(f)
            S = pickle.load(f)

    return K, S


# class Results :
# 	def __init__(self, sep, K, S, Sc, pivot, clustering) :


if __name__ == '__main__':

    K = {}
    S = {}
    C = {}
    Stab = {}
    S_Stab = {}
    sep = 2

    for prob in range(1):
        bin = binpacking(
            prob_num=prob, type = 'u1000', sep = 2
        )

        print(prob)

        # bin.Kelly()
        # K[prob] = bin.Kelly_result
        #
        # print('Kelly')
        #
        # bin.Separation()
        # S[prob] = bin.Separation_result
        #
        # bin.chevy()
        # C[prob] = bin.chevy_result
        #
        # bin.Stabilization()
        # Stab[prob] = bin.Stab_Result

        bin.Sep_Stab()
        S_Stab[prob] = bin.Sep_Stab_Result



    clustering = False
    # save_results(K, S, sep, clustering)

    # K,S = read_results(sep, clustering)

    print(result_table2(S_Stab, C))
# print(obj_change_graph(problems, K, S, sep))
# print(iteration_graph(problems, K, S, sep))
