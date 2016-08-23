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

def FindMaximalRobustness(optimization_problem_type, args):
  input_id = args.input_id
  data_dir = args.data_dir

  print("\nProcess Weights")
  weights = np.genfromtxt(data_dir+'/weights3.csv', delimiter=',')

  in_num, out_num = weights.shape #32, 10

  W = np.transpose(weights)
  threshold = 1e-1
  WPruned = np.array([ [ 0 if w <= threshold and w >= -threshold else w for w in row] for row in W ])
  #print(W[0])
  #print(WPruned[0])

  print("Process Biases")
  B = np.genfromtxt(data_dir+'/biases3.csv', delimiter=',')
  BPruned = np.array([ 0 if b <= threshold and b >= -threshold else b for b in B ])

  solvers = [ pywraplp.Solver('FindMaximalRobustness%d' % i,
              optimization_problem_type) for i in range(out_num * (out_num - 1)) ]

  global_min_upper = sys.maxint
  for k in range(out_num):
    solver = solvers[k]
    infinity = solver.infinity()

    #print("Forward Calculation")
    In = [ solver.NumVar(1.0, infinity, 'x%s' % i) for i in range(in_num) ]
    Out = vmmul(In, W, B, in_num, out_num)
    OutPruned = vmmul(In, WPruned, BPruned, in_num, out_num)

    # Only minimize makes sense, because upper is the max bound, and maxmax doesn't make sense. minmax does. Similarly for maxmin.
    #print("Creating Constraints")
    upper = solver.NumVar(0.0, infinity, 'upper')
    solver.Minimize(upper)
    #lower = solver.NumVar(0.0, infinity, 'lower')
    #solver.Maximize(lower)
    #solver.Minimize(upper-lower)
    #upper = [ solver.NumVar(0.0, 1.0, 'u%s' % i) for i in range(in_num) ]
    #solver.Minimize(sum(upper))
    #lower = [ solver.NumVar(0.0, 1.0, 'l%s' % i) for i in range(in_num) ]
    #solver.Maximize(sum(lower))
    #region = [u - l for u, l in zip(upper, lower)]
    #solver.Minimize(sum(region))

    # Create bounds constraints
    #bound_cond = [ lower[i] <= upper[i] for i in range(in_num) ]
    #bound_cond = lower <= upper

    # Create input layer constraints
    #input_cond1 = [ In[i] <= upper[i] for i in range(in_num) ]
    #input_cond2 = [ In[i] >= lower[i] for i in range(in_num) ]
    input_cond1 = [ In[i] <= upper for i in range(in_num) ]
    #input_cond2 = [ In[i] >= lower for i in range(in_num) ]

    # Create output layer constraints
    maxId = k
    output_cond = [ (Out[maxId] >= Out[i] + 1e-2) for i in range(out_num) if i != maxId ]
    output_pruned_cond = [ (OutPruned[maxId] <= OutPruned[i] + 1e-2) for i in range(out_num) if i != maxId ]

    # Add all constraints
    for j in range (in_num):
      solver.Add(input_cond1[j])
      #solver.Add(input_cond2[j])

    for j in range(out_num - 1):
      solver.Add(output_cond[j])

    min_upper = sys.maxint
    for j in range(out_num - 1):
      solver.Add(output_pruned_cond[j])

      #print("Start solving with %d constraints %d variables" % (solver.NumConstraints(), solver.NumVariables()))
      result_status = solver.Solve()

      if (result_status == pywraplp.Solver.OPTIMAL):
        # The solution looks legit (when using solvers others than
        # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
        if (solver.VerifySolution(1e-7, True)):
          #print(('Problem solved in %f milliseconds' % solver.wall_time()))
          # The objective value of the solution.
          result = solver.Objective().Value()
          if (result < min_upper):
            min_upper = result
          #print('Optimal objective value = %f [%f, %f]' % (result, lower.solution_value(), upper.solution_value()))
          #print('%d %d Optimal objective value = %f' % (k, j, result))
      else:
        print(LP_Solution_Status[result_status])

    if (min_upper < global_min_upper):
      global_min_upper = min_upper
  
  print('%d Optimal objective value = %f' % (k, min_upper))
  return global_min_upper

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

  FindMaximalRobustness(pywraplp.Solver.GLOP_LINEAR_PROGRAMMING, args)


if __name__ == '__main__':
  main()
