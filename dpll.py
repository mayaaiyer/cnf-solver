import copy
import random
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

vars = set()

def parse_einstein():
    with open('einstein_input.cnf', 'r') as f:
        output = f.read().splitlines()
        cnf = list()
        cnf.append(list())
        num_of_var = 0

        for line in output:
            tokens = line.split()
            if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                for tok in tokens:
                    lit = int(tok)
                    num_of_var = max(num_of_var, abs(lit))
                    if lit == 0:
                        cnf.append(list())
                    else:
                        cnf[-1].append(lit)

        assert len(cnf[-1]) == 0
        cnf.pop()
        return cnf, num_of_var

def maya_heuristic(cnf, assignments):
    jw_dict = {}
    for clause in cnf:
        for prop in clause:
            if (prop in jw_dict):
                jw_dict[prop] += 2.0 ** (-1 * len(clause))
            else:
                jw_dict[prop] = 2.0 ** (-1 * len(clause))

    prop = None

    if len(jw_dict) == 0:
        left_to_assign = list(vars - set(np.absolute(assignments)))
        prop = random.choice(left_to_assign)
    else:
        prop = list(jw_dict.keys())[list(jw_dict.values()).index(max(jw_dict.values()))]
    
    return prop

def random_heuristic(cnf, assignments):
    left_to_assign = list(vars - set(np.absolute(assignments)))
    prop = random.choice(left_to_assign)
    return prop

def two_clause_heuristic(cnf, assignments):
    freq = {}
    for clause in cnf:
        if len(clause) == 2:
            if (clause[0] in freq):
                freq[clause[0]] += 1
            else:
                freq[clause[0]] = 1
            
            if (clause[1] in freq):
                freq[clause[1]] += 1
            else:
                freq[clause[1]] = 1

    prop = None

    if len(freq) == 0:
        left_to_assign = list(vars - set(np.absolute(assignments)))
        prop = random.choice(left_to_assign)
    else:
        prop = list(freq.keys())[list(freq.values()).index(max(freq.values()))]
    
    return prop

def dpll(heuristic, i, cnf, assignments):
    i += 1
    # unit-preference -> make assignments
    unit_preference(cnf, assignments)

    if cnf == None or len(cnf) == 0:
        return True, assignments, i
    if len(min(cnf, key=len)) == 0:
        return False, assignments, i

    # splitting rule
    prop = heuristic(cnf, assignments)

    temp_assignments = copy.deepcopy(assignments)
    temp_cnf = copy.deepcopy(cnf)
    temp_cnf.append([prop])

    boolean, temp_assignments, i = dpll(heuristic, i, temp_cnf, temp_assignments)
    if boolean:
        return True, temp_assignments, i
    
    temp_assignments = copy.deepcopy(assignments)
    temp_cnf = copy.deepcopy(cnf)
    temp_cnf.append([-prop])
    
    return dpll(heuristic, i, temp_cnf, temp_assignments)
    
def unit_preference(cnf, assignments):
    while len(min(cnf, key=len)) == 1:
        #there is a clause of length 1
        unit_clause = min(cnf, key=len)
        if -unit_clause[0] in assignments: # add termination condition if the assignment is invalid
            cnf.append([])
        elif not unit_clause[0] in assignments: # assign if not already assigned
            assignments.append(unit_clause[0])
        cnf.remove(unit_clause)
        if len(cnf) == 0:
            return

    # simplify on assignments

    for assignment in assignments:
        for clause in cnf[:]: # remove all instances of the assignment proposition
            if assignment in clause:
                cnf.remove(clause)
            if (-1 * assignment) in clause:
                clause.remove(-1 * assignment)

def generate_solution_einstein(boolean, final_assignments, clauses, num_of_var, elapsed_time, i):
    assignments = [("v " + str(x)) for x in final_assignments]

    solution = os.linesep.join(assignments)

    solution_variable = os.linesep.join(["s cnf" + (" 1 " if boolean else " 0 ") + str(num_of_var) + " " + str(clauses), solution])
    solution_comment = os.linesep.join(["c Einstein's puzzle's solution", solution_variable])
    solution_final = os.linesep.join(["t cnf" + (" 1 " if boolean else " 0 ") + str(num_of_var) + " " + str(clauses) + " " + str(elapsed_time) + " " + str(i), solution_comment])
    
    return solution_final

def generate_random_model(L=450, N=150):
    variables = range(1, N + 1)
    cnf = list()
    for i in range(int(L)):
        propositions = random.sample(variables, 3)
        for j in range(len(propositions)):
            if random.random() > 0.5:
                propositions[j] = propositions[j] * -1
        cnf.append(propositions)
    return cnf

if __name__ == '__main__':
    random.seed(7102003)
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-hr", "--heuristic", help="the heuristic to run (random, two-clause, maya's)")
    argParser.add_argument("-t", "--test", help="the method of test (random-model, einstein)")
    args = argParser.parse_args()

    if args.test == "einstein":     
        cnf, num_of_var = parse_einstein()
        clauses = len(cnf)
        vars = set(range(1, num_of_var + 1))
        assignments = list()
        iter = 0
        boolean, final_assignments = None, None
        start_time = time.time()
        if args.heuristic == "two-clause": # run two-clause heuristic
            boolean, final_assignments, iter = dpll(two_clause_heuristic, iter, cnf, assignments)
        elif args.heuristic == "maya's": # run maya's heuristic
            boolean, final_assignments, iter = dpll(maya_heuristic, iter, cnf, assignments)
        else: # run random
            boolean, final_assignments, iter = dpll(random_heuristic, iter, cnf, assignments)

        end_time = time.time()
        elapsed_time = end_time - start_time

        final_assignments.sort()

        with open('einstein_output.cnf', 'w') as f:
            solution = generate_solution_einstein(boolean, final_assignments, clauses, num_of_var, elapsed_time, iter)
            f.write(solution)
    else:
        tests = 100
        # run experiments and generate plots
        x_axis = np.asarray([3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])

        prob_sat_N1_mi = list()
        median_time_N1_mi = list()
        median_iterations_N1_mi = list()

        prob_sat_N1_random = list()
        median_time_N1_random = list()
        median_iterations_N1_random = list()

        prob_sat_N1_tc = list()
        median_time_N1_tc = list()
        median_iterations_N1_tc = list()

        prob_sat_N2_mi = list()
        median_time_N2_mi = list()
        median_iterations_N2_mi = list()

        prob_sat_N2_random = list()
        median_time_N2_random = list()
        median_iterations_N2_random = list()

        prob_sat_N2_tc = list()
        median_time_N2_tc = list()
        median_iterations_N2_tc = list()

        # run experiments
        for ratio in x_axis:
            N = 150
            L = ratio * N
            vars = set(range(1, N + 1))
            sat_mi = np.zeros(tests)
            time_arr_mi = np.zeros(tests)
            iterations_mi = np.zeros(tests)

            sat_random = np.zeros(tests)
            time_arr_random = np.zeros(tests)
            iterations_random = np.zeros(tests)

            sat_tc = np.zeros(tests)
            time_arr_tc = np.zeros(tests)
            iterations_tc = np.zeros(tests)

            #N1
            for i in range(tests):
                # generate cnf
                cnf_temp = generate_random_model(L, N)

                # run with maya heuristic
                cnf = copy.deepcopy(cnf_temp)
                assignments = list()
                iter = 0
                boolean, final_assignments = None, None
                start_time = time.time()
                boolean, final_assignments, iter = dpll(maya_heuristic, iter, cnf, assignments)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if boolean:
                    sat_mi[i] = 1
                else:
                    sat_mi[i] = 0
                time_arr_mi[i] = elapsed_time
                iterations_mi[i] = iter

                # run with random heuristic
                cnf = copy.deepcopy(cnf_temp)
                assignments = list()
                iter = 0
                boolean, final_assignments = None, None
                start_time = time.time()
                boolean, final_assignments, iter = dpll(random_heuristic, iter, cnf, assignments)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if boolean:
                    sat_random[i] = 1
                else:
                    sat_random[i] = 0
                time_arr_random[i] = elapsed_time
                iterations_random[i] = iter

                # run with two clause heuristic
                cnf = copy.deepcopy(cnf_temp)
                assignments = list()
                iter = 0
                boolean, final_assignments = None, None
                start_time = time.time()
                boolean, final_assignments, iter = dpll(two_clause_heuristic, iter, cnf, assignments)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if boolean:
                    sat_tc[i] = 1
                else:
                    sat_tc[i] = 0
                time_arr_tc[i] = elapsed_time
                iterations_tc[i] = iter


            prob_sat_N1_mi.append(np.mean(sat_mi))
            median_time_N1_mi.append(np.median(time_arr_mi))
            median_iterations_N1_mi.append(np.median(iterations_mi))

            prob_sat_N1_random.append(np.mean(sat_random))
            median_time_N1_random.append(np.median(time_arr_random))
            median_iterations_N1_random.append(np.median(iterations_random))

            prob_sat_N1_tc.append(np.mean(sat_tc))
            median_time_N1_tc.append(np.median(time_arr_tc))
            median_iterations_N1_tc.append(np.median(iterations_tc))

            N = 200
            L = ratio * N
            vars = set(range(1, N + 1))

            #N2
            for i in range(tests):
                # generate cnf
                cnf_temp = generate_random_model(L, N)

                # run with maya heuristic
                cnf = copy.deepcopy(cnf_temp)
                assignments = list()
                iter = 0
                boolean, final_assignments = None, None
                start_time = time.time()
                boolean, final_assignments, iter = dpll(maya_heuristic, iter, cnf, assignments)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if boolean:
                    sat_mi[i] = 1
                else:
                    sat_mi[i] = 0
                time_arr_mi[i] = elapsed_time
                iterations_mi[i] = iter

                # run with random heuristic
                cnf = copy.deepcopy(cnf_temp)
                assignments = list()
                iter = 0
                boolean, final_assignments = None, None
                start_time = time.time()
                boolean, final_assignments, iter = dpll(random_heuristic, iter, cnf, assignments)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if boolean:
                    sat_random[i] = 1
                else:
                    sat_random[i] = 0
                time_arr_random[i] = elapsed_time
                iterations_random[i] = iter

                # run with two clause heuristic
                cnf = copy.deepcopy(cnf_temp)
                assignments = list()
                iter = 0
                boolean, final_assignments = None, None
                start_time = time.time()
                boolean, final_assignments, iter = dpll(two_clause_heuristic, iter, cnf, assignments)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if boolean:
                    sat_tc[i] = 1
                else:
                    sat_tc[i] = 0
                time_arr_tc[i] = elapsed_time
                iterations_tc[i] = iter

            prob_sat_N2_mi.append(np.mean(sat_mi))
            median_time_N2_mi.append(np.median(time_arr_mi))
            median_iterations_N2_mi.append(np.median(iterations_mi))

            prob_sat_N2_random.append(np.mean(sat_random))
            median_time_N2_random.append(np.median(time_arr_random))
            median_iterations_N2_random.append(np.median(iterations_random))

            prob_sat_N2_tc.append(np.mean(sat_tc))
            median_time_N2_tc.append(np.median(time_arr_tc))
            median_iterations_N2_tc.append(np.median(iterations_tc))

        # generate plots

        fig, ax = plt.subplots()
        ax.plot(x_axis, prob_sat_N1_mi, label ='N = 150')
        ax.plot(x_axis, prob_sat_N2_mi, label ='N = 200')
        plt.title('probability of satisfiability vs L/N (maya\'s heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("probability of satisfiability")
        plt.legend()
        plt.savefig('images/sat_mi.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, prob_sat_N1_random, label ='N = 150')
        ax.plot(x_axis, prob_sat_N2_random, label ='N = 200')
        plt.title('probability of satisfiability vs L/N (random heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("probability of satisfiability")
        plt.legend()
        plt.savefig('images/sat_random.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, prob_sat_N1_tc, label ='N = 150')
        ax.plot(x_axis, prob_sat_N2_tc, label ='N = 200')
        plt.title('probability of satisfiability vs L/N (two-clause heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("probability of satisfiability")
        plt.legend()
        plt.savefig('images/sat_tc.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, median_time_N1_mi, label ='N = 150')
        ax.plot(x_axis, median_time_N2_mi, label ='N = 200')
        plt.title('median run time vs L/N (maya\'s heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("median run time")
        plt.legend()
        plt.savefig('images/time_mi.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, median_time_N1_random, label ='N = 150')
        ax.plot(x_axis, median_time_N2_random, label ='N = 200')
        plt.title('median run time vs L/N (random heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("median run time")
        plt.legend()
        plt.savefig('images/time_random.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, median_time_N1_tc, label ='N = 150')
        ax.plot(x_axis, median_time_N2_tc, label ='N = 200')
        plt.title('median run time vs L/N (two-clause heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("median run time")
        plt.legend()
        plt.savefig('images/time_tc.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, median_iterations_N1_mi, label ='N = 150')
        ax.plot(x_axis, median_iterations_N2_mi, label ='N = 200')
        plt.title('num of DPLL calls vs L/N (maya\'s heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("num of DPLL calls")
        plt.legend()
        plt.savefig('images/iter_mi.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, median_iterations_N1_random, label ='N = 150')
        ax.plot(x_axis, median_iterations_N2_random, label ='N = 200')
        plt.title('num of DPLL calls vs L/N (random heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("num of DPLL calls")
        plt.legend()
        plt.savefig('images/iter_random.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, median_iterations_N1_tc, label ='N = 150')
        ax.plot(x_axis, median_iterations_N2_tc, label ='N = 200')
        plt.title('num of DPLL calls vs L/N (two-clause heuristic)')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("num of DPLL calls")
        plt.legend()
        plt.savefig('images/iter_tc.png')

        maya_v_random_N1_time = []
        for i in range(len(median_time_N1_mi)):
            maya_v_random_N1_time.append(median_time_N1_mi[i] / median_time_N1_random[i])
        maya_v_random_N1_iter = []
        for i in range(len(median_iterations_N1_mi)):
            maya_v_random_N1_iter.append(median_iterations_N1_mi[i] / median_iterations_N1_random[i])
        maya_v_random_N2_time = []
        for i in range(len(median_time_N2_mi)):
            maya_v_random_N2_time.append(median_time_N2_mi[i] / median_time_N2_random[i])
        maya_v_random_N2_iter = []
        for i in range(len(median_iterations_N2_mi)):
            maya_v_random_N2_iter.append(median_iterations_N2_mi[i] / median_iterations_N2_random[i])
            
        maya_v_tc_N1_time = []
        for i in range(len(median_time_N1_mi)):
            maya_v_tc_N1_time.append(median_time_N1_mi[i] / median_time_N1_tc[i])
        maya_v_tc_N1_iter = []
        for i in range(len(median_iterations_N1_mi)):
            maya_v_tc_N1_iter.append(median_iterations_N1_mi[i] / median_iterations_N1_tc[i])
        maya_v_tc_N2_time = []
        for i in range(len(median_time_N2_mi)):
            maya_v_tc_N2_time.append(median_time_N2_mi[i] / median_time_N2_tc[i])
        maya_v_tc_N2_iter = []
        for i in range(len(median_iterations_N2_mi)):
            maya_v_tc_N2_iter.append(median_iterations_N2_mi[i] / median_iterations_N2_tc[i])

        fig, ax = plt.subplots()
        ax.plot(x_axis, maya_v_random_N1_time, label ='N = 150')
        ax.plot(x_axis, maya_v_random_N2_time, label ='N = 200')
        plt.title('ratio of run time (maya/random) vs L/N')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("maya's heuristic run time / random heuristic run time")
        plt.legend()
        plt.savefig('images/mayavrandom_time.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, maya_v_random_N1_iter, label ='N = 150')
        ax.plot(x_axis, maya_v_random_N2_iter, label ='N = 200')
        plt.title('ratio of DPLL calls (maya/random) vs L/N')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("maya's heuristic DPLL calls / random heuristic DPLL calls")
        plt.legend()
        plt.savefig('images/mayavrandom_iter.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, maya_v_tc_N1_time, label ='N = 150')
        ax.plot(x_axis, maya_v_tc_N2_time, label ='N = 200')
        plt.title('ratio of run time (maya/two-clause) vs L/N')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("maya's heuristic run time / two-clause heuristic run time")
        plt.legend()
        plt.savefig('images/mayavtc_time.png')

        fig, ax = plt.subplots()
        ax.plot(x_axis, maya_v_tc_N1_iter, label ='N = 150')
        ax.plot(x_axis, maya_v_tc_N2_iter, label ='N = 200')
        plt.title('ratio of DPLL calls (maya/two-clause) vs L/N')
        plt.xlabel("L/N (L = |cnf|, N = |Prop|)")
        plt.ylabel("maya's heuristic DPLL calls / two-clause heuristic DPLL calls")
        plt.legend()
        plt.savefig('images/mayavtc_iter.png')



