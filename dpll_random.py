import copy
import random
import os
import time
import argparse
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

def dpll_random(i, cnf, assignments):
    i += 1
    # unit-preference -> make assignments
    unit_preference(cnf, assignments)

    if cnf == None or len(cnf) == 0:
        return True, assignments, i
    if len(min(cnf, key=len)) == 0:
        return False, assignments, i

    # splitting rule
    left_to_assign = list(vars - set(np.absolute(assignments)))
    prop = random.choice(left_to_assign)

    temp_assignments = copy.deepcopy(assignments)
    temp_cnf = copy.deepcopy(cnf)
    temp_cnf.append([prop])

    boolean, temp_assignments, i = dpll_random(i, temp_cnf, temp_assignments)
    if boolean:
        return True, temp_assignments, i
    
    temp_assignments = copy.deepcopy(assignments)
    temp_cnf = copy.deepcopy(cnf)
    temp_cnf.append([-prop])
    
    return dpll_random(i, temp_cnf, temp_assignments)
    

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

def generate_solution(boolean, final_assignments, clauses, num_of_var, elapsed_time, i):
    assignments = [("v " + str(x)) for x in final_assignments]

    solution = os.linesep.join(assignments)

    solution_variable = os.linesep.join(["s cnf" + (" 1 " if boolean else " 0 ") + str(num_of_var) + " " + str(clauses), solution])
    solution_comment = os.linesep.join(["c Einstein's puzzle's solution", solution_variable])
    solution_final = os.linesep.join(["t cnf" + (" 1 " if boolean else " 0 ") + str(num_of_var) + " " + str(clauses) + " " + str(elapsed_time) + " " + str(i), solution_comment])
    
    return solution_final


if __name__ == '__main__':
    random.seed(7102003)
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-hr", "--heuristic", help="the heuristic to run (random, two-clause, maya's)")

    args = argParser.parse_args()
    cnf, num_of_var = parse_einstein()
    clauses = len(cnf)
    vars = set(range(1, num_of_var + 1))
    assignments = list()
    i = 0
    
    boolean, final_assignments = None, None

    start_time = time.time()
    if args.heuristic == "two-clause": # run two-clause heuristic
        boolean, final_assignments, i = dpll_random(i, cnf, assignments)
    elif args.heuristic == "maya's": # run maya's heuristic
        boolean, final_assignments, i = dpll_random(i, cnf, assignments)
    else: # run random
        boolean, final_assignments, i = dpll_random(i, cnf, assignments)

    end_time = time.time()
    elapsed_time = end_time - start_time

    final_assignments.sort()

    with open('einstein_output.cnf', 'w') as f:
        solution = generate_solution(boolean, final_assignments, clauses, num_of_var, elapsed_time, i)
        f.write(solution)

