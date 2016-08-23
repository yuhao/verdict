from __future__ import print_function
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver import pywraplp
import numpy as np
import argparse
import sys

LP_Solution_Status = ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED", "ABNORMAL", "NOT_SOLVED"]

def vvmul(V1, V2, bias, n):
  res = 0.0
  for i in range(0, n):
    res += V1[i] * V2[i]
  res += bias
  return res

def vmmul(V, M, B, m, n):
  res = [None] * n
  for i in range(0, n):
    res[i] = vvmul(V, M[i], B[i], m)
  return res

def FindRange(bound, base, in_max, W, WPruned, B, BPruned, in_num, out_num, optimization_problem_type):
  solvers = [ pywraplp.Solver('FindMaximalRange%d' % i,
              optimization_problem_type) for i in range(out_num * 2) ]

  min_upper = in_max
  for k in range(len(solvers)):
    solver = solvers[k]
    infinity = solver.infinity()

    #print("Forward Calculation")
    In = [ solver.NumVar(base, in_max, 'x%s' % i) for i in range(in_num) ]
    Out = vmmul(In, W, B, in_num, out_num)
    OutPruned = vmmul(In, WPruned, B, in_num, out_num)

    #print("Creating Constraints")
    upper = solver.NumVar(0.0, in_max, 'upper')
    solver.Minimize(upper)

    # Create input layer constraints
    input_cond = [ In[i] <= upper for i in range(in_num) ]

    # Create output layer constraints
    output_cond = [ Out[i] >= bound for i in range(out_num) ] + [ OutPruned[i] >= bound for i in range(out_num) ]

    # Add all constraints
    for j in range (in_num):
      solver.Add(input_cond[j])

    solver.Add(output_cond[k])

    result_status = solver.Solve()

    if (result_status == pywraplp.Solver.OPTIMAL):
      if (solver.VerifySolution(1e-7, True)):
        #print(('Problem solved in %f milliseconds' % solver.wall_time()))
        result = solver.Objective().Value()
        if (result < min_upper):
          min_upper = result
        #print('%d Optimal objective value = %f' % (k, result))
    else:
      print(k, LP_Solution_Status[result_status])

  return min_upper


def FindConstraints(optimization_problem_type, args):
  input_id = args.input_id
  data_dir = args.data_dir

  print("\nProcess Parameters")
  weights = np.genfromtxt(data_dir+'/weights2.csv', delimiter=',')
  B = np.genfromtxt(data_dir+'/biases2.csv', delimiter=',')

  in_num, out_num = weights.shape

  W = np.transpose(weights)
  threshold = 1e-1
  WPruned = np.array([ [ 0 if w <= threshold and w >= -threshold else w for w in row] for row in W ])
  BPruned = np.array([ 0 if b <= threshold and b >= -threshold else b for b in B ])
  print("Pruned weights:", np.count_nonzero(W) - np.count_nonzero(WPruned))

  inMax = float("inf")
  nextUpper = 0.0
  upper = 0.0
  step = 0
  while (upper == 0.0 or nextUpper - upper > 1e-4):
    upper = nextUpper
    nextUpper = FindRange(1.046903, upper, inMax, W, WPruned, B, BPruned, in_num, out_num, optimization_problem_type)
    step += 1
    print("%d [%f, %f]" % (step, upper, nextUpper))


def main():
  parser = argparse.ArgumentParser(description="Find Maximal Perturbation a Model Can Tolerate.")
  parser.add_argument("-c", "--input-id",
                      dest="input_id",
                      type=int,
                      help="input id",
                      default=0)
  parser.add_argument("-d", "--data-dir",
                      dest="data_dir",
                      help="directory for model parameters",
                      default="../mnist/para")
  args = parser.parse_args()

  FindConstraints(pywraplp.Solver.GLOP_LINEAR_PROGRAMMING, args)


if __name__ == '__main__':
  main()
