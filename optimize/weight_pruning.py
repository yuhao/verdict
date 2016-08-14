from __future__ import print_function
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver import pywraplp
import numpy as np
import argparse

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

def FindMaximalPruning(optimization_problem_type, args):
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
  Input = [inputs[input_id], inputs[input_id + 1], inputs[input_id + 2]]

  g_L1X = []
  g_L1Y = []
  g_L2X = []
  g_L2Y = []
  g_OutX = []
  g_OutY = []
  g_maxId = []
  g_hl1_sign = []
  g_hl2_sign = []
  g_hl1_cond = []
  g_hl2_cond = []
  g_output_cond = []

  for i in range(len(Input)):
    InX = Input[i]
    L1X, hl1_sign = vmmul_produce_sign(InX, W1, B1, l0_n, l1_n, True)
    L2X, hl2_sign = vmmul_produce_sign(L1X, W2, B2, l1_n, l2_n, True)
    OutX, _ = vmmul_produce_sign(L2X, W3, B3, l2_n, l3_n, False)
    maxId = np.argmax(OutX)
    g_L1X.append(L1X)
    g_L2X.append(L2X)
    g_OutX.append(OutX)
    g_maxId.append(maxId)
    g_hl1_sign.append(hl1_sign)
    g_hl2_sign.append(hl2_sign)

  solver = pywraplp.Solver('FindMaximalPruning', optimization_problem_type)
  infinity = solver.infinity()

  print("Create Constraints")
  W1_flag = [ [ solver.IntVar(0, 1, 'w1-%s-%s' %(j, i)) for j in range(l0_n) ] for i in range(l1_n) ]
  W1_pruned = [ [ W1_flag[i][j] * W1[i][j] for j in range(l0_n) ] for i in range(l1_n) ]

  sum_flags = solver.IntVar(0, l0_n * l1_n, 'sum_flags')
  sum_flags = sum([sum(w) for w in W1_flag])

  solver.Minimize(sum_flags)

  # Create hidden layer constraints
  for i in range(len(Input)):
    InX = Input[i]
    L1Y, hl1_cond = vmmul_consume_sign(InX, g_hl1_sign[i], W1_pruned, B1, l0_n, l1_n, True)
    L2Y, hl2_cond = vmmul_consume_sign(L1Y, g_hl2_sign[i], W2, B2, l1_n, l2_n, True)
    OutY, _ = vmmul_consume_sign(L2Y, None, W3, B3, l2_n, l3_n, False)
    g_L1Y.append(L1Y)
    g_L2Y.append(L2Y)
    g_OutY.append(OutY)
    g_hl1_cond.append(hl1_cond)
    g_hl2_cond.append(hl2_cond)

  # Create output layer constraints
  #g_output_cond = [ [ (g_OutY[k][i] + 0.01 <= g_OutY[k][maxId]) for i in range(l3_n) if i != maxId ] for k in range(len(Input)) ]
  for k in range(len(Input)):
    maxId = g_maxId[k]
    output_cond = [ (g_OutY[k][i] + 0.01 <= g_OutY[k][maxId]) for i in range(l3_n) if i != maxId ]
    g_output_cond.append(output_cond)

  print("Add Constraints")
  for i in range(len(Input)):
    for j in range(l1_n):
      solver.Add(g_hl1_cond[i][j])

    for j in range(l2_n):
      solver.Add(g_hl2_cond[i][j])

    for j in range(l3_n - 1):
      solver.Add(g_output_cond[i][j])

  print("Start solving with %d constraints %d variables" % (solver.NumConstraints(), solver.NumVariables()))
  result = SolveAndPrint(solver)

  if (result == pywraplp.Solver.OPTIMAL):
      min_sum = solver.Objective().Value()
      #OutY_Value = [ OutY[i].solution_value() for i in range(l3_n)]
      #W1_flag_value = [ [ W1_flag[i][j].solution_value() for j in range(l0_n)] for i in range(l1_n) ]
    #print("Max Sum:", min_sum)
  else:
    print(LP_Solution_Status[result])

  #print('W1_flag[0] =', W1_flag_value[0])
  #print('W1[0] =', W1[0])
  #print('OutX =', OutX)
  #print('OutY =', OutY_Value)
  #print('argmax(OutX) =', np.argmax(OutX))
  #print('argmax(OutY) =', np.argmax(OutY_Value))

  return l0_n * l1_n, min_sum

def SolveAndPrint(solver):
  """Solve the problem and print the solution."""

  result_status = solver.Solve()

  if (result_status == pywraplp.Solver.OPTIMAL):
    # The solution looks legit (when using solvers others than
    # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
    if (solver.VerifySolution(1e-7, True)):
      print(('Problem solved in %f milliseconds' % solver.wall_time()))
      # The objective value of the solution.
      print(('Optimal objective value = %f' % solver.Objective().Value()))

  return result_status


def main():
  parser = argparse.ArgumentParser(description="Find Maximal Weights Pruning a Model Can Tolerate.")
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

  total, retained = FindMaximalPruning(pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING, args)
  print("\nMaximal Tolerable Weights Pruning for Image %d is %ld (%f)" % (args.input_id, total - retained, 1 - retained / total))


if __name__ == '__main__':
  main()
