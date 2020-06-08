import pandas as pd
import numpy as np
import cplex
import time
import matplotlib.pyplot as plt
import csv
import pickle
import ctypes
import sys
from sklearn.cluster import KMeans
import math

knapsack = ctypes.CDLL('knapsack.so')
knapsack.knapsack_bnb.restype = ctypes.c_double


class Logger:
    def __init__(self, filename):
        self.f = open(filename, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.f is None:
            f = self.f
            self.f = None
            f.close()

    def flush(self):
        if not self.f is None:
            self.f.flush()
            sys.stdout.flush()

    def write(self, text):
        if not self.f is None:
            self.f.write(text)
            sys.stdout.write(text)


class GAP:
    def __init__(self, problem='d10100', sep=3, timelimit=60, clustering=True, RMP=False):

        self.problem = problem
        self.ty = problem[0]
        self.sep = sep
        self.timelimit = timelimit
        self.clustering = clustering
        self.RMP = RMP
        self.data()

    def data(self):
        ty = self.ty
        problem = self.problem

        f = open("gap_%s/%s.txt" % (ty, problem), 'r')
        data = f.readlines()
        f.close()

        records = []
        for line in data:
            record = [int(field) for field in line.strip().lstrip('[').rstrip(']').split()]
            records.append(record)

        size = records[0]
        agent = size[0]
        job = size[1]
        # print(job)
        c = []
        a = []
        b = []
        for i in range(len(records) - 1):
            if len(c) < job * agent:
                c.extend(records[i + 1])
            elif len(c) >= job * agent and len(a) < job * agent:
                a.extend(records[i + 1])
            else:
                b.extend(records[i + 1])

        self.c = np.array(c, dtype=int).reshape((agent, job))
        self.a = np.array(a, dtype=int).reshape((agent, job))
        self.b = np.array(b)

        self.agent = agent
        self.job = job

    def CPLEX(self):

        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c
        self.RMP = False
        RMP = self.RMP


        mt = 0
        tt = 0

        ct = time.time()

        M = cplex.Cplex()

        x_i_j = lambda i, j: 'x_%d_%d' % (i, j)
        x = [x_i_j(i, j) for i in range(agent) for j in range(job)]

        M.variables.add(
            lb=[0] * len(x),
            ub=[1] * len(x),
            names=x,
            obj=[int(c[i][j]) for i in range(agent) for j in range(job)],
            types=['B'] * len(x)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(i, j) for i in range(agent)],
                    val=[1] * agent
                )
                for j in range(job)
            ],
            senses=["E" for j in range(job)],
            rhs=[1.0 for j in range(job)])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(i, j) for j in range(job)],
                    val=[int(a[i][j]) for j in range(job)]
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[int(b[i]) for i in range(agent)])

        M.objective.set_sense(M.objective.sense.minimize)
        M.parameters.timelimit.set(self.timelimit)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        at = time.time()
        #
        # if RMP == False:
        #
        #     with Logger('Results/cplex/%s_%s.txt' % (self.problem, self.timelimit)) as logger:
        #         M.set_results_stream(logger)
        #         M.solve()
        # else:
        M.set_problem_type(M.problem_type.LP)
        M.solve()
        self.getRMP = list(M.solution.get_dual_values())

        mt += time.time() - at


        obj = M.solution.get_objective_value()
        x_val = M.solution.get_values(x)

        tt += time.time() - ct

        self.CPLEX_result = [

            'CPLEX', mt, tt, obj, M

        ]

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

    def Kelly(self):
        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c

        iterations = []

        K = range(1)
        var = list(range(agent))

        M = cplex.Cplex()
        M.parameters.lpmethod.set(1)

        x_i_k = lambda i, k: 'x_%d_%d' % (i, k)
        x = [x_i_k(i, k) for i in range(1) for k in K]

        dummy = float(sum(np.sum(c, axis=1)))

        M.variables.add(
            lb=[0] * len(x),
            ub=[1] * len(x),
            names=x,
            obj=[dummy],
            types=['C'] * len(x)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(i, k) for i in range(1) for k in K],
                    val=[1.0]
                )
                for j in range(job)
            ],
            senses=["G" for j in range(job)],
            rhs=[1.0 for j in range(job)],
            names=['assignment_%d' % (j) for j in range(job)])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(0, k) for k in K],
                    val=[0] * len(K)
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[1.0 for i in range(agent)])

        for ag in range(agent):
            w = list(a[ag])
            v = list(np.array(c[ag]))
            W = int(b[ag])

            S_obj, sol = GAP.KnapsackBnB(v, w, W)

            newsub = sol
            idx = M.variables.get_num()
            M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
            M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                           [idx] * job,
                                                           newsub)))

            M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
            var.append(idx)

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        start = time.time()
        mt = 0
        st = 0
        ite = 0
        solutions = {}
        criteria = [True] * agent

        while any(criteria):

            ite += 1

            M.set_problem_type(M.problem_type.LP)
            ct = time.time()

            if ite == 1:
                self.RMP = True
                self.CPLEX()
                dual = self.getRMP

            else:
                M.solve()
                solutions[time.time()-start]=float(M.solution.get_objective_value())
                iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

                mt += time.time() - ct

                dual = list(M.solution.get_dual_values())

            pi = list(dual)[:job]
            # dual = list(M.solution.get_dual_values())

            if ite % 1000 == 0:
                print(ite)

            for ag in range(agent):

                w = list(a[ag])
                v = list(np.array(pi) - np.array(c[ag]))
                W = int(b[ag])

                pt = time.time()

                S_obj, sol = GAP.KnapsackBnB(v, w, W)

                st += time.time() - pt

                if S_obj - 0.00001 > -dual[job + ag]:
                    # if 100 < time.time()-start :
                    #     self.sols.append(M.solution.get_objective_value())
                    #     print(time.time()-start, M.solution.get_objective_value() )

                    if time.time()-start > self.timelimit:
                        print(time.time()-start)
                        M.set_problem_type(M.problem_type.LP)
                        ct = time.time()
                        M.solve()
                        solutions[time.time()-start]=float(M.solution.get_objective_value())
                        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
                        mt += time.time() - ct
                        tt = time.time() - start

                        self.Kelly_result = [
                            'Kelly', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))
                        ]

                        return


                    criteria[ag] = True
                    newsub = sol
                    idx = M.variables.get_num()
                    M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
                    M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                   [idx] * job,
                                                                   newsub)))

                    M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
                    var.append(idx)
                else:
                    criteria[ag] = False

        M.set_problem_type(M.problem_type.LP)
        ct = time.time()
        M.solve()
        solutions[time.time()-start]=float(M.solution.get_objective_value())
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        mt += time.time() - ct
        tt = time.time() - start

        # print(var)

        self.Kelly_M = M

        self.Kelly_result = [

            'Kelly', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]



    def Kelly_CG(self, M, tolerence):
        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c


        mt = 0
        st = 0
        ite = 0
        solutions = []
        iterations = []
        criteria = [True] * agent

        while all(criteria):

            ite += 1

            M.set_problem_type(M.problem_type.LP)
            ct = time.time()

            M.solve()
            solutions.append(float(M.solution.get_objective_value()))
            iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

            mt += time.time() - ct

            dual = list(M.solution.get_dual_values())

            pi = list(dual)[:job]
            # dual = list(M.solution.get_dual_values())

            for ag in range(agent):

                w = list(a[ag])
                v = list(np.array(pi) - np.array(c[ag]))
                W = int(b[ag])

                pt = time.time()

                S_obj, sol = GAP.KnapsackBnB(v, w, W)

                st += time.time() - pt

                if S_obj - tolerence > -dual[job + ag]:
                    criteria[ag] = True
                    newsub = sol
                    idx = M.variables.get_num()
                    M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
                    M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                   [idx] * job,
                                                                   newsub)))

                    M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
                    # var.append(idx)
                else:
                    criteria[ag] = False
                    break

        M.set_problem_type(M.problem_type.LP)
        ct = time.time()
        M.solve()
        solutions.append(float(M.solution.get_objective_value()))
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        self.Kelly_M = M


        Kelly_result = [

            ite, mt, st,  solutions, iterations

        ]

        return Kelly_result


    def Stabilization(self):

        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c

        K = range(1)
        var = list(range(agent))
        eps = 0.1

        M = cplex.Cplex()

        x_i_k = lambda i, k: 'x_%d_%d' % (i, k)
        x = [x_i_k(i, k) for i in range(1) for k in K]

        dummy = float(sum(np.sum(c, axis=1)))

        M.variables.add(
            lb=[0] * len(x),
            ub=[1] * len(x),
            names=x,
            obj=[dummy],
            types=['C'] * len(x)
        )

        gp_j = lambda j: 'gp_%d' % (j)

        gp = [gp_j(j) for j in range(job)]

        M.variables.add(
            lb=[0] * len(gp),
            ub=[eps] * len(gp),
            names=gp,
            obj=[0] * len(gp),
            types=['C'] * len(gp)
        )

        gm_j = lambda j: 'gm_%d' % (j)

        gm = [gm_j(j) for j in range(job)]

        M.variables.add(
            lb=[0] * len(gm),
            ub=[eps] * len(gm),
            names=gm,
            obj=[0] * len(gm),
            types=['C'] * len(gm)
        )

        yp_i = lambda i: 'yp_%d' % (i)

        yp = [yp_i(i) for i in range(agent)]

        M.variables.add(
            lb=[0] * len(yp),
            ub=[eps] * len(yp),
            names=yp,
            obj=[0] * len(yp),
            types=['C'] * len(yp)
        )

        ym_i = lambda i: 'ym_%d' % (i)

        ym = [ym_i(i) for i in range(agent)]

        M.variables.add(
            lb=[0] * len(ym),
            ub=[eps] * len(ym),
            names=ym,
            obj=[0] * len(ym),
            types=['C'] * len(ym)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [gp_j(j)] + [gm_j(j)],
                    val=[1.0] + [1.0, -1.0]
                )
                for j in range(job)
            ],
            senses=["G" for j in range(job)],
            rhs=[1.0 for j in range(job)],
            names=['assignment_%d' % (j) for j in range(job)])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [yp_i(i)] + [ym_i(i)],
                    val=[0] * len(K) + [1.0] + [-1.0]
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[1.0 for i in range(agent)])

        M.objective.set_sense(M.objective.sense.minimize)
        M.write('first.lp')

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
        criteria = [True] * agent

        while any(criteria):

            ite += 1

            M.set_problem_type(M.problem_type.LP)

            ct = time.time()
            M.solve()
            solutions.append(float(M.solution.get_objective_value()))
            iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
            mt += time.time() - ct

            dual = list(M.solution.get_dual_values())
            pi = dual[:job]
            phi = dual[job:]

            for ag in range(agent):

                w = list(a[ag])
                v = list(np.array(pi) - np.array(c[ag]))
                W = int(b[ag])

                pt = time.time()

                S_obj, sol = GAP.KnapsackBnB(v, w, W)

                st += time.time() - pt

                if (S_obj - 0.000001 > -dual[job + ag]) or eps != 0:

                    criteria[ag] = True

                    if ite % 30 == 1 :

                        M.objective.set_linear(
                            zip(gp + gm + yp + ym, pi + list(-np.array(pi)) + phi + list(-np.array(phi))))

                    newsub = sol
                    idx = M.variables.get_num()
                    M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
                    M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                   [idx] * job,
                                                                   newsub)))

                    M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
                    var.append(idx)

                    if ite % 100 == 0:
                        eps *= 0.1
                        if ite == 300:
                            eps = 0

                        for dv in gm + gp + ym + yp:
                            M.variables.set_upper_bounds(dv, eps)



                else:
                    criteria[ag] = False

        M.set_problem_type(M.problem_type.LP)
        ct = time.time()
        M.write('last.lp')
        M.solve()
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        solutions.append(float(M.solution.get_objective_value()))
        mt += time.time() - ct
        tt = time.time() - start

        self.Stabilization_result = [

            'Stabilization', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]


    def Cheby(self):

        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c
        self.alpha = 1
        K = range(1)
        var = list(range(agent))
        eps = 0.1

        M = cplex.Cplex()

        x_i_k = lambda i, k: 'x_%d_%d' % (i, k)
        x = [x_i_k(i, k) for i in range(1) for k in K]

        dummy = float(sum(np.sum(c, axis=1)))

        M.variables.add(
            lb=[0] * len(x),
            # ub=[1] * len(x),
            names=x,
            obj=[dummy*100],
            types=['C'] * len(x)
        )

        u_j = lambda j: 'u_%d' % (j)

        u = [u_j(j) for j in range(job)]

        M.variables.add(
            lb=[0] * len(u),
            # ub=[eps] * len(u),
            names=u,
            obj=[0] * len(u),
            types=['C'] * len(u)
        )

        v_i = lambda i: 'v_%d' % (i)

        v = [v_i(i) for i in range(agent)]

        M.variables.add(
            lb=[0] * len(v),
            # ub=[eps] * len(gm),
            names=v,
            obj=[0] * len(v),
            types=['C'] * len(v)
        )
        aa = [0] * job

        for i in range(job):
            aa[i] = float(min(c.T[i]))

        M.variables.add(
            lb=[0],
            # ub=[eps] * len(gm),
            names='z',
            obj=[-sum(aa)],
            types=['C']
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [u[j]] + ['z'],
                    val=[1.0]*len(x) + [-1.0, -1.0]
                )
                for j in range(job)
            ],
            senses=["G" for j in range(job)],
            rhs=[0 for j in range(job)],
            names=['assignment_%d' % (j) for j in range(job)])


        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [v_i(i)] + ['z'],
                    val=[1.0]*len(x)+ [1.0] + [-1.0]
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[0 for i in range(agent)])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + u + v + ['z'],
                    val=[math.sqrt(float(len(x)+agent)+1.0)]*len(x)+ [1.0]*len(u) + [1.0]*len(v) + [self.alpha*math.sqrt(float(agent+job))]
                )
            ],
            senses=["G"],
            rhs=[1])
        #
        # for ag in range(agent):
        #     w = list(a[ag])
        #     v = list(np.array(c[ag]))
        #     W = int(b[ag])
        #
        #     S_obj, sol = GAP.KnapsackBnB(v, w, W)
        #
        #     newsub = sol
        #
        #     idx = M.variables.get_num()
        #     M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
        #     M.linear_constraints.set_coefficients(list(zip(list(range(job)),
        #                                                    [idx] * job,
        #                                                    newsub)))
        #
        #     M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
        #     var.append(idx)

        M.objective.set_sense(M.objective.sense.minimize)
        # M.write('first.lp')

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
        criteria = [True] * agent
        # Z_ =



        while True:
            ite += 1
            M.set_problem_type(M.problem_type.LP)

            ct = time.time()

            M.solve()

            z_sol = M.solution.get_values(['z'])[0]
            obj_sol = M.solution.get_objective_value()

            # print(z_sol,obj_sol)

            solutions.append(float(obj_sol))
            iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
            mt += time.time() - ct

            dual = list(M.solution.get_dual_values())
            pi = dual[:job]
            phi = dual[job:job+agent]

            print('intro')
            print(dual[-1],z_sol)
            #
            # if ite == 5 :
            #     M.write('cheby.lp')
            #     return

            if dual[-1] > 0.0001 * z_sol :
                print('if')

                criteria = [True] * agent
                for ag in range(agent):

                    w = list(a[ag])
                    v = list(np.array(pi) - np.array(c[ag]))
                    W = int(b[ag])
                    pt = time.time()

                    S_obj, sol = GAP.KnapsackBnB(v, w, W)

                    st += time.time() - pt



                    if (S_obj - 0.000001 > -dual[job + ag]):

                        print(ite,ag)
                        newsub = sol
                        idx = M.variables.get_num()
                        M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
                        M.linear_constraints.set_coefficients(list(zip(list(range(job))+[job+agent],
                                                                       [idx] * (job+1),
                                                                       newsub+[math.sqrt(sum([i**2 for i in newsub])+1)])))

                        M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
                        var.append(idx)
                        criteria[ag] = True

                    else :
                        criteria[ag] = False

                if any(criteria) == False:
                    print('dd',criteria)

                    Z_ = sum(pi)- sum(phi)

                    M.objective.set_linear(
                        zip(['z'],  [-Z_]))


            else:
                M.write('cheby.lp')
                break
                # pass
                # criteria[ag] = False

        M.set_problem_type(M.problem_type.LP)
        self.Z = Z_
        ct = time.time()
        # M.write('last.lp')
        M.solve()
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        solutions.append(float(M.solution.get_objective_value()))
        mt += time.time() - ct
        tt = time.time() - start
        self.Cheby_M = M
        self.Cheby_result = [

            'Cheby', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]


    def Sep_Stabilization(self):

        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c
        sep = self.sep

        K = range(1)
        var = list(range(agent))
        J = list(range(job))
        Js = list(np.array_split(J, sep))

        M = cplex.Cplex()
        M.parameters.lpmethod.set(2)

        x_i_k = lambda i, k: 'x_%d_%d' % (i, k)
        x = [x_i_k(i, k) for i in range(1) for k in K]

        dummy = float(sum(np.sum(c, axis=1)))

        M.variables.add(
            lb=[0] * len(x),
            ub=[1] * len(x),
            names=x,
            obj=[dummy],
            types=['C'] * len(x)
        )

        y_j = lambda j: 'y_%d' % (j)

        y = [y_j(j) for j in J]

        ys = [[y[i] for i in Js[j]] for j in range(sep)]

        M.variables.add(
            lb=[-cplex.infinity] * len(y),
            ub=[cplex.infinity] * len(y),
            names=y,
            obj=[0] * len(y),
            types=['C'] * len(y)
        )

        ######

        eps = 0.1

        gp_j = lambda j: 'gp_%d' % (j)

        gp = [gp_j(j) for j in range(job)]

        M.variables.add(
            lb=[0] * len(gp),
            ub=[eps] * len(gp),
            names=gp,
            obj=[0] * len(gp),
            types=['C'] * len(gp)
        )

        gm_j = lambda j: 'gm_%d' % (j)

        gm = [gm_j(j) for j in range(job)]

        M.variables.add(
            lb=[0] * len(gm),
            ub=[eps] * len(gm),
            names=gm,
            obj=[0] * len(gm),
            types=['C'] * len(gm)
        )

        yp_i = lambda i: 'yp_%d' % (i)

        yp = [yp_i(i) for i in range(agent)]

        M.variables.add(
            lb=[0] * len(yp),
            ub=[eps] * len(yp),
            names=yp,
            obj=[0] * len(yp),
            types=['C'] * len(yp)
        )

        ym_i = lambda i: 'ym_%d' % (i)

        ym = [ym_i(i) for i in range(agent)]

        M.variables.add(
            lb=[0] * len(ym),
            ub=[eps] * len(ym),
            names=ym,
            obj=[0] * len(ym),
            types=['C'] * len(ym)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [gp_j(j)] + [gm_j(j)] + [y_j(j)],
                    val=[1.0] + [1.0, -1.0] + [1.0]
                )
                for j in range(job)
            ],
            senses=["G" for j in range(job)],
            rhs=[1.0 for j in range(job)],
            names=['assignment_%d' % (j) for j in range(job)])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=x + [yp_i(i)] + [ym_i(i)],
                    val=[0] * len(K) + [1.0] + [-1.0]
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[1.0 for i in range(agent)])

        M.objective.set_sense(M.objective.sense.minimize)
        M.write('first.lp')

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
        criteria = [True] * agent


        for sec in range(sep):

            criteria = [True] * agent

            while any(criteria):

                ite += 1

                M.set_problem_type(M.problem_type.LP)

                # y_fix = list(set(y) - set(list(ys[sec])))
                # J_fix = list(set(J) - set(list(Js[sec])))

                y_fix = list(ys[sec])
                J_fix = list(Js[sec])
                ### Set y = 0 for all J (same as original problem)

                # if repeat == 0:
                M.variables.set_upper_bounds(zip(y_fix, [0] * len(y_fix)))
                M.variables.set_lower_bounds(zip(y_fix, [0] * len(y_fix)))


                # else:
                #     M.variables.set_upper_bounds(zip(y, [0] * len(y)))
                #     M.variables.set_lower_bounds(zip(y, [0] * len(y)))

                ct = time.time()

                ### Set initial RMP to the optimal solution of LP relaxation problem

                if ite == 1:
                    self.RMP = True
                    self.CPLEX()
                    dual = self.getRMP

                else:

                    M.solve()
                    solutions.append(float(M.solution.get_objective_value()))
                    iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

                    mt += time.time() - ct

                    dual = list(M.solution.get_dual_values())

                    pi = dual[:job]
                    pi_ = [dual[i] for i in J_fix]
                    phi = dual[job:]
                    # print(pi_[:10])


                    for ag in range(agent):

                        w = list(a[ag])
                        v = list(np.array(pi) - np.array(c[ag]))
                        W = int(b[ag])

                        pt = time.time()

                        S_obj, sol = GAP.KnapsackBnB(v, w, W)

                        st += time.time() - pt

                        if S_obj - 0.00001 > -dual[job + ag] or eps != 0:
                            criteria[ag] = True

                            if ite % 30  == 1:
                                M.objective.set_linear(
                                    zip(gp + gm + yp + ym, pi + list(-np.array(pi)) + phi + list(-np.array(phi))))

                                M.objective.set_linear(zip(y_fix, pi_))
                            newsub = sol
                            idx = M.variables.get_num()
                            M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
                            M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                           [idx] * job,
                                                                           newsub)))

                            M.linear_constraints.set_coefficients(job + ag, idx, 1.0)

                            var.append(idx)

                            if ite % 100 == 0:
                                eps *= 0.1
                                if ite == 300:
                                    eps = 0

                                for dv in gm + gp + ym + yp:
                                    M.variables.set_upper_bounds(dv, eps)

                        else:
                            criteria[ag] = False



        M.set_problem_type(M.problem_type.LP)
        M.variables.set_upper_bounds(zip(y, [0] * len(y)))
        M.variables.set_lower_bounds(zip(y, [0] * len(y)))

        ct = time.time()
        # M.write('test.lp')
        M.solve()
        solutions.append(float(M.solution.get_objective_value()))
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        #
        #
        # Kelly_result = self.Kelly_CG(M)
        #
        # ite += Kelly_result[0]
        # mt += Kelly_result[1]
        # st += Kelly_result[2]
        # solutions += Kelly_result[3]
        # iterations += Kelly_result[4]

        mt += time.time() - ct
        tt = time.time() - start
        self.Sep_Stab_M = M

        self.Sep_Stab_result = [

            'sep+stab', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

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

    @staticmethod
    def Master_Kelly(agent, job, a, b, c):

        K = range(1)
        var = list(range(agent))

        M = cplex.Cplex()
        M.parameters.lpmethod.set(1)

        x_i_k = lambda i, k: 'x_%d_%d' % (i, k)
        x = [x_i_k(i, k) for i in range(1) for k in K]

        dummy = float(sum(np.sum(c, axis=1)))

        M.variables.add(
            lb=[0] * len(x),
            ub=[1] * len(x),
            names=x,
            obj=[dummy],
            types=['C'] * len(x)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(i, k) for i in range(1) for k in K],
                    val=[1.0]
                )
                for j in range(job)
            ],
            senses=["G" for j in range(job)],
            rhs=[1.0 for j in range(job)],
            names=['assignment_%d' % (j) for j in range(job)])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(0, k) for k in K],
                    val=[0] * len(K)
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[1.0 for i in range(agent)])

        for ag in range(agent):
            w = list(a[ag])
            v = list(np.array(c[ag]))
            W = int(b[ag])

            S_obj, sol = GAP.KnapsackBnB(v, w, W)

            newsub = sol
            idx = M.variables.get_num()
            M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
            M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                           [idx] * job,
                                                           newsub)))

            M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
            var.append(idx)

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        return M, var

    @staticmethod
    def Master_Sep(agent, job, a, b, c, sep, J, Js, var):

        M = cplex.Cplex()
        M.parameters.lpmethod.set(2)
        K = range(1)

        x_i_k = lambda i, k: 'x_%d_%d' % (i, k)
        x = [x_i_k(i, k) for i in range(1) for k in K]

        dummy = float(sum(np.sum(c, axis=1)))

        M.variables.add(
            lb=[0] * len(x),
            ub=[1] * len(x),
            names=x,
            obj=[dummy],
            types=['C'] * len(x)
        )

        y_j = lambda j: 'y_%d' % (j)

        y = [y_j(j) for j in J]

        ys = [[y[i] for i in Js[j]] for j in range(sep)]

        M.variables.add(
            lb=[-cplex.infinity] * len(y),
            ub=[cplex.infinity] * len(y),
            names=y,
            obj=[0] * len(y),
            types=['C'] * len(y)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(i, k) for i in range(1) for k in K] + [y_j(j)],
                    val=[1.0] + [1.0]
                )
                for j in J
            ],
            senses=["G" for j in J],
            rhs=[1.0 for j in J],
            names=['assignment_%d' % (j) for j in J])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(0, k) for k in K],
                    val=[0] * len(K)
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[1.0 for i in range(agent)])

        for ag in range(agent):
            w = list(a[ag])
            v = list(np.array(c[ag]))
            W = int(b[ag])

            S_obj, sol = GAP.KnapsackBnB(v, w, W)

            newsub = sol
            idx = M.variables.get_num()
            M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
            M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                           [idx] * job,
                                                           newsub)))

            M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
            var.append(idx)

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        return M, var, y, ys

    def Separation_CG(self):

        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c
        sep = self.sep

        var = list(range(agent))
        J = list(range(job))

        if self.clustering == True:
            Js = GAP.cluster(sep, a.reshape(job, agent), c.reshape(job, agent), J)

        else:
            Js = list(np.array_split(J, sep))

        MM, M_var = GAP.Master_Kelly(agent, job, a, b, c)

        for sec in range(self.sep):
            M, var, y, ys = GAP.Master_Sep(agent, job, a, b, c, sep, J, Js, var)
            print(self.ite)
            criteria = [True] * agent

            while any(criteria):

                self.ite += 1

                M.set_problem_type(M.problem_type.LP)

                y_fix = list(set(y) - set(list(ys[sec])))
                J_fix = list(set(J) - set(list(Js[sec])))

                ### Set y = 0 for all J (same as original problem)

                M.variables.set_upper_bounds(zip(y_fix, [0] * len(y_fix)))
                M.variables.set_lower_bounds(zip(y_fix, [0] * len(y_fix)))

                ct = time.time()

                ### Set initial RMP to the optimal solution of LP relaxation problem

                if self.ite == 1:
                    self.RMP = True
                    self.CPLEX()
                    dual = self.getRMP

                else:
                    M.solve()
                    self.solutions.append(float(M.solution.get_objective_value()))
                    self.iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

                    self.mt += time.time() - ct

                    dual = list(M.solution.get_dual_values())

                pi = dual[:job]
                pi_ = [dual[i] for i in J_fix]

                for ag in range(agent):

                    w = list(a[ag])
                    v = list(np.array(pi) - np.array(c[ag]))
                    W = int(b[ag])

                    pt = time.time()

                    S_obj, sol = GAP.KnapsackBnB(v, w, W)

                    self.st += time.time() - pt

                    if S_obj - 0.00001 > -dual[job + ag]:

                        criteria[ag] = True

                        M.objective.set_linear(zip(y_fix, pi_))
                        newsub = sol
                        label = int('%d000' % (sec + 1))
                        idx = M.variables.get_num() + label
                        # print(idx)
                        M.variables.add(names=['x_%d' % (idx)], obj=[float(np.array(sol).T @ c[ag])])
                        M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                       ['x_%d' % (idx)] * job,
                                                                       newsub)))

                        M.linear_constraints.set_coefficients(job + ag, idx - label, 1.0)
                        var.append(idx)

                        ### For Kelly Master Problem
                        # idx = MM.variables.get_num()
                        MM.variables.add(names=['x_%d' % (idx)], obj=[float(np.array(sol).T @ c[ag])])
                        MM.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                        ['x_%d' % (idx)] * job,
                                                                        newsub)))
                        MM.linear_constraints.set_coefficients(job + ag, 'x_%d' % (idx), 1.0)

                        M_var.append(idx)


                    else:
                        criteria[ag] = False

        self.M = MM

    def Separation_new(self):

        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c
        sep = self.sep

        start = time.time()
        self.mt = 0
        self.st = 0
        self.ite = 0
        self.solutions = []
        self.iterations = []

        GAP.Separation_CG(self)

        print(self.ite)

        M = self.M

        criteria = [True] * agent

        while any(criteria):

            self.ite += 1

            M.set_problem_type(M.problem_type.LP)
            ct = time.time()
            M.solve()
            self.solutions.append(float(M.solution.get_objective_value()))
            self.iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

            self.mt += time.time() - ct

            dual = list(M.solution.get_dual_values())

            pi = list(dual)[:job]

            for ag in range(agent):

                w = list(a[ag])
                v = list(np.array(pi) - np.array(c[ag]))
                W = int(b[ag])

                pt = time.time()

                S_obj, sol = GAP.KnapsackBnB(v, w, W)

                self.st += time.time() - pt

                if S_obj - 0.00001 > -dual[job + ag]:

                    criteria[ag] = True
                    newsub = sol
                    idx = M.variables.get_num()
                    M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
                    M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                   [idx] * job,
                                                                   newsub)))

                    M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
                    # var.append(idx)
                else:
                    criteria[ag] = False

        M.set_problem_type(M.problem_type.LP)
        ct = time.time()
        M.solve()
        self.solutions.append(float(M.solution.get_objective_value()))
        self.iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))
        self.mt += time.time() - ct
        tt = time.time() - start

        self.Separation_M = M

        self.Separation_result = [

            'Separation', self.ite, self.mt, self.st, tt, self.mt / (self.st + self.mt), self.solutions,
            np.average(np.array(self.iterations))

        ]

    def Separation(self):
        self.sep_sols = []

        agent = self.agent
        job = self.job
        a = self.a
        b = self.b
        c = self.c
        sep = self.sep

        K = range(1)
        var = list(range(agent))
        J = list(range(job))

        if self.clustering == True:
            Js = GAP.cluster(sep, a.reshape(job, agent), c.reshape(job, agent), J)

        else:
            Js = list(np.array_split(J, sep))

        M = cplex.Cplex()
        M.parameters.lpmethod.set(2)

        x_i_k = lambda i, k: 'x_%d_%d' % (i, k)
        x = [x_i_k(i, k) for i in range(1) for k in K]

        dummy = float(sum(np.sum(c, axis=1)))

        M.variables.add(
            lb=[0] * len(x),
            ub=[1] * len(x),
            names=x,
            obj=[dummy],
            types=['C'] * len(x)
        )

        y_j = lambda j: 'y_%d' % (j)

        y = [y_j(j) for j in J]

        ys = [[y[i] for i in Js[j]] for j in range(sep)]

        M.variables.add(
            lb=[-cplex.infinity] * len(y),
            ub=[cplex.infinity] * len(y),
            names=y,
            obj=[0] * len(y),
            types=['C'] * len(y)
        )

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(i, k) for i in range(1) for k in K] + [y_j(j)],
                    val=[1.0] + [1.0]
                )
                for j in J
            ],
            senses=["G" for j in J],
            rhs=[1.0 for j in J],
            names=['assignment_%d' % (j) for j in J])

        M.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_k(0, k) for k in K],
                    val=[0] * len(K)
                )
                for i in range(agent)
            ],
            senses=["L" for i in range(agent)],
            rhs=[1.0 for i in range(agent)])

        for ag in range(agent):
            w = list(a[ag])
            v = list(np.array(c[ag]))
            W = int(b[ag])

            S_obj, sol = GAP.KnapsackBnB(v, w, W)

            newsub = sol
            idx = M.variables.get_num()
            M.variables.add(obj=[float(np.array(sol).T @ c[ag])])
            M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                           [idx] * job,
                                                           newsub)))

            M.linear_constraints.set_coefficients(job + ag, idx, 1.0)
            var.append(idx)

        M.objective.set_sense(M.objective.sense.minimize)
        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)

        start = time.time()
        mt = 0
        st = 0
        ite = 0
        solutions = {}

        iterations = []

        #### Type 1 :
        print('--------- START ---------')

        alpha = 1

        for repeat in range(2):

        # while True:

            for sec in range(sep):
                print(ite)
                criteria = [True] * agent

                while any(criteria):

                    ite += 1
                    if ite % 100 == 0 :
                        alpha = alpha*0.8
                    elif alpha >= 500:
                        alpha = 1




                    M.set_problem_type(M.problem_type.LP)

                    y_fix = list(set(y) - set(list(ys[sec])))
                    J_fix = list(set(J) - set(list(Js[sec])))

                    ### Set y = 0 for all J (same as original problem)

                    # if repeat == 0:
                    M.variables.set_upper_bounds(zip(y_fix, np.zeros(len(y_fix))))
                    M.variables.set_lower_bounds(zip(y_fix, [0] * len(y_fix)))


                    # else:
                    #     M.variables.set_upper_bounds(zip(y, [0] * len(y)))
                    #     M.variables.set_lower_bounds(zip(y, [0] * len(y)))

                    ct = time.time()

                    ### Set initial RMP to the optimal solution of LP relaxation problem

                    if ite == 1:
                        self.RMP = True
                        self.CPLEX()
                        dual = self.getRMP

                    else:

                        M.solve()
                        solutions[time.time()-start]=float(M.solution.get_objective_value())
                        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

                        mt += time.time() - ct

                        dual = list(M.solution.get_dual_values())

                        pi = dual[:job]
                        pi_ = [alpha * dual[i] for i in J_fix]
                        # print(pi_[:10])




                        for ag in range(agent):

                            w = list(a[ag])
                            v = list(np.array(pi) - np.array(c[ag]))
                            W = int(b[ag])

                            pt = time.time()

                            S_obj, sol = GAP.KnapsackBnB(v, w, W)

                            st += time.time() - pt

                            if S_obj - 0.000001 > -dual[job + ag]:



                                criteria[ag] = True


                                M.objective.set_linear(zip(y_fix, pi_))

                                newsub = sol
                                idx = M.variables.get_num()
                                M.variables.add(obj=[float(np.array(sol).T @ c[ag])])


                                M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                               [idx] * job,
                                                                               newsub)))

                                M.linear_constraints.set_coefficients(job + ag, idx, 1.0)

                                var.append(idx)

                            else:
                                criteria[ag] = False


        print('--------- Finished ---------')
        M.set_problem_type(M.problem_type.LP)
        # M.variables.set_upper_bounds(zip(y, [0] * len(y)))
        # M.variables.set_lower_bounds(zip(y, [0] * len(y)))

        ct = time.time()

        M.solve()
        solutions[time.time()-start]=float(M.solution.get_objective_value())
        iterations.append(float(cplex._internal._procedural.getitcnt(M._env._e, M._lp)))

        #
        # Kelly_result = self.Kelly_CG(M)
        #
        # ite += Kelly_result[0]
        # mt += Kelly_result[1]
        # st += Kelly_result[2]
        # solutions += Kelly_result[3]
        # iterations += Kelly_result[4]

        mt += time.time() - ct
        tt = time.time() - start
        self.Separation_M = M

        self.Separation_result = [

            'Separation', ite, mt, st, tt, mt / (st + mt), solutions, np.average(np.array(iterations))

        ]


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
    result = pd.concat([k, s], axis=1)
    result.to_csv('basic.csv')
    return result


def result_table2(results):
    # results = [pd.DataFrame(S), pd.DataFrame(K)]
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

    import cProfile



    K = {}
    S = {}
    C = {}
    CS = {}
    Stab = {}
    sep = 2
    clustering = False
    problems = ['d10100']
    # problems = ['d05100','d10100','d10200','d20100','d20200','e05100','e10100','e10200','e20100','e20200']
    # problems = ['d10400','e10400']
    for problem in problems:
        gap = GAP(
            problem=problem, sep=sep, timelimit=3600, clustering=clustering
        )

        gap.Cheby()
        # C[problem] = gap.CPLEX_result
        CS[problem] = gap.Cheby_result

        # gap.Kelly()
        # K[problem] = gap.Kelly_result
        # print(problem,'M',gap.Kelly_M.solution.get_objective_value())

        # print(problem,'K')
        # gap.Separation_new()
        # gap.Separation()
        # S[problem] = gap.Separation_result
        # print(problem,'S',gap.Separation_M.solution.get_objective_value())

        # gap.Sep_Stabilization()
        # S[problem] = gap.Sep_Stab_result
        # print(problem,'S',gap.Sep_Stab_M.solution.get_objective_value())

        # gap.Stabilization()

    # Stab[problem] = gap.Stabilization_result

    # save_results(K, S, sep, clustering)

    # K,S = read_results(sep, clustering)

    # with open('Results/result_Twice.txt', 'wb') as f:
    #     pickle.dump(K, f)
    #     pickle.dump(S, f)

    print(result_table2([pd.DataFrame(CS), pd.DataFrame(K)]))



# print(K[problem][-2][-1])
# print(S[problem][-2][-1])


# print(obj_change_graph(problems, K, S, sep))
# print(iteration_graph(problems, K, S, sep))


