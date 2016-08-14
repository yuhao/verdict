from __future__ import print_function
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver import pywraplp
import numpy as np
import argparse
import time

LP_Solution_Status = ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED", "ABNORMAL", "NOT_SOLVED"]

def vvmul(V1, V2, bias, n):
  res = 0.0
  for i in range(0, n):
    res += V1[i] * V2[i]
  res += bias
  return res

def vmmul_produce_sign(V, M, B, m, n, act):
  res = [None] * n
  sign = [None] * n
  for i in range(0, n):
    tmp = vvmul(V, M[i], B[i], m)
    if act == True:
      res[i] = tmp if(tmp > 0) else 0
      sign[i] = True if(tmp > 0) else False
    else:
      res[i] = tmp
  return (res, sign)

def vmmul_consume_sign(V, S, M, B, m, n, act):
  res = [None] * n
  cond = [None] * n
  for i in range(0, n):
    tmp = vvmul(V, M[i], B[i], m)

    if act == True:
      if (S[i] == True):
        cond[i] = (-tmp <= 0)
        res[i] = tmp
      else:
        cond[i] = (tmp <= 0) #It's OK to use <=
        res[i] = 0
    else:
      res[i] = tmp
  return (res, cond)

def FindMaximalRobustness(optimization_problem_type, args):
  input_id = args.input_id
  data_dir = args.data_dir

  print("\nProcess Weights")
  weights1 = np.genfromtxt(data_dir+'/weights1.csv', delimiter=',')
  weights2 = np.genfromtxt(data_dir+'/weights2.csv', delimiter=',')
  weights3 = np.genfromtxt(data_dir+'/weights3.csv', delimiter=',')

  l0_n, l1_n = weights1.shape #784, 128
  l2_n, l3_n = weights3.shape #32, 10

  W1 = np.transpose(weights1)
  W2 = np.transpose(weights2)
  W3 = np.transpose(weights3)

  print("Process Biases")
  B1 = np.genfromtxt(data_dir+'/biases1.csv', delimiter=',')
  B2 = np.genfromtxt(data_dir+'/biases2.csv', delimiter=',')
  B3 = np.genfromtxt(data_dir+'/biases3.csv', delimiter=',')

  print("Process Reference Input")
  inputs = np.genfromtxt('../mnist/para/mnist_test_images_100.csv', delimiter=',')
  InX = inputs[input_id]
  L1X, hl1_sign = vmmul_produce_sign(InX, W1, B1, l0_n, l1_n, True)
  L2X, hl2_sign = vmmul_produce_sign(L1X, W2, B2, l1_n, l2_n, True)
  OutX, _ = vmmul_produce_sign(L2X, W3, B3, l2_n, l3_n, False)
  maxId = np.argmax(OutX)

  solvers = [ pywraplp.Solver('FindMaximalRobustness%d' % i,
                  optimization_problem_type) for i in range(l3_n - 1) ]
  min_delta = 1.0
  start_time = time.time()
  for s in range(l3_n - 1):
    solver = solvers[s]
    infinity = solver.infinity()

    print("\nIteration %d: Creating Constraints" % s)
    W1_prime = [ [ solver.NumVar(-infinity, infinity, 'y%s-%s' %(j, i)) for j in range(l0_n) ] for i in range(l1_n) ]
    delta = solver.NumVar(0.0, 1.0, 'delta')

    solver.Minimize(delta)

    # Create weight constraints
    w1_cond = [None] * l0_n * l1_n * 2
    for i in range (l1_n):
      for j in range (l0_n):
        w1_cond[i * l0_n + j] = (W1[i][j] - W1_prime[i][j] <= delta)
        w1_cond[l0_n * l1_n + i * l0_n + j] = (W1_prime[i][j] - W1[i][j] <= delta)

    # Create hidden layer constraints
    L1Y, hl1_cond = vmmul_consume_sign(InX, hl1_sign, W1_prime, B1, l0_n, l1_n, True)
    L2Y, hl2_cond = vmmul_consume_sign(L1Y, hl2_sign, W2, B2, l1_n, l2_n, True)
    OutY, _ = vmmul_consume_sign(L2Y, None, W3, B3, l2_n, l3_n, False)

    # Create output layer constraints
    output_cond = [ (OutY[maxId] <= OutY[i] + 0.01) for i in range(l3_n) if i != maxId ]

    # Add all constraints
    for i in range (l1_n):
      for j in range (l0_n):
        solver.Add(w1_cond[i * l0_n + j])
        solver.Add(w1_cond[l0_n * l1_n + i * l0_n + j])

    for j in range(l1_n):
      solver.Add(hl1_cond[j])

    for j in range(l2_n):
      solver.Add(hl2_cond[j])

    solver.Add(output_cond[s])

    print("Start solving with %d constraints %d variables" % (solver.NumConstraints(), solver.NumVariables()))
    result = SolveAndPrint(solver)

    if (result == pywraplp.Solver.OPTIMAL):
      if (solver.Objective().Value() < min_delta):
        min_delta = solver.Objective().Value()
        #OutY_Value = [ OutY[i].solution_value() for i in range(l3_n)]
      #print("Min Delta:", min_delta)
    else:
      print(LP_Solution_Status[result])

  elapsed_time = time.time() - start_time
  print(elapsed_time)
  #print('OutX =', OutX)
  #print('OutY =', OutY_Value)
  #print('argmax(OutX) =', np.argmax(OutX))
  #print('argmax(OutY) =', np.argmax(OutY_Value))

  return min_delta, elapsed_time

def SolveAndPrint(solver):
  """Solve the problem and print the solution."""

  result_status = solver.Solve()

  if (result_status == pywraplp.Solver.OPTIMAL):
    # The solution looks legit (when using solvers others than
    # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
    if (solver.VerifySolution(1e-7, True)):
      #print(('Problem solved in %f milliseconds' % solver.wall_time()))
      # The objective value of the solution.
      print(('Optimal objective value = %f' % solver.Objective().Value()))

  return result_status


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

  max_robustness, duration = FindMaximalRobustness(pywraplp.Solver.GLOP_LINEAR_PROGRAMMING, args)
  print("\nMaximal Tolerable Weights Variance for Image %d is %f (%f s)" % (args.input_id, max_robustness, duration))


if __name__ == '__main__':
  main()
