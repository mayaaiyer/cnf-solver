import copy
import random
import os
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

def dpll(cnf, assignments):
    #unit-preference -> make assignments
    unit_preference(cnf, assignments)

    if cnf == None or len(cnf) == 0:
        return True, assignments
    if len(min(cnf, key=len)) == 0:
        return False, assignments

    #splitting rule
    left_to_assign = list(vars - set(np.absolute(assignments)))
 
    prop = random.choice(left_to_assign)

    temp_assignments = copy.deepcopy(assignments)
    temp_cnf = copy.deepcopy(cnf)
    temp_cnf.append([prop])

    boolean, temp_assignments = dpll(temp_cnf, temp_assignments)
    if boolean:
        return True, temp_assignments
    
    temp_assignments = copy.deepcopy(assignments)
    temp_cnf = copy.deepcopy(cnf)
    temp_cnf.append([-prop])
    
    return dpll(temp_cnf, temp_assignments)
    

def unit_preference(cnf, assignments):
    while len(min(cnf, key=len)) == 1:
        #there is a clause of length 1
        unit_clause = min(cnf, key=len)
        assignments.append(unit_clause[0])
        while(unit_clause in cnf): #remove all instances of the unit clause
            cnf.remove(unit_clause)
        for clause in cnf: #remove all instances of the assignment proposition
            if unit_clause[0] in clause:
                cnf.remove(clause)
            if (-unit_clause[0]) in clause:
                clause.remove(-unit_clause[0])
        if len(cnf) == 0:
            return

    #simplify on assignments
    for assignment in assignments:
        for clause in cnf: #remove all instances of the assignment proposition
            if assignment in clause:
                cnf.remove(clause)
            if (-assignment) in clause:
                clause.remove(-assignment)

def generate_solution(boolean, final_assignments, clauses, num_of_var):
    assignments = [("v " + str(x)) for x in final_assignments]

    solution = os.linesep.join(assignments)

    solution_variable = os.linesep.join(["s cnf" + " 1 " if boolean else " 0 " + str(num_of_var) + " " + str(clauses), solution])
    solution_final = os.linesep.join(["c Einstein's puzzle's solution", solution_variable])

    return solution_final


if __name__ == '__main__':
    cnf, num_of_var = parse_einstein()
    clauses = len(cnf)
    vars = set(range(1, num_of_var + 1))
    assignments = list()
    boolean, final_assignments = dpll(cnf, assignments)

    final_assignments.sort()

    with open('einstein_output.cnf', 'w') as f:
        solution = generate_solution(boolean, final_assignments, clauses, num_of_var)
        f.write(solution)

