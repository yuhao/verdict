# Copyright (c) Microsoft Corporation 2015

#inputL(l0) --> hiddenL1(l1) --> hiddenL2(l2) --> outputL(l3)

from z3 import *
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Verdict harness.")
parser.add_argument("-i", "--input-perturbation",
                    dest="input_pert",
                    type=float,
                    help="input perturbation. choose between (0, 1)",
                    default=0.0001)
parser.add_argument("-r", "--robustness",
                    dest="robust_cons",
                    help="robustness constraint. choose between <weak, medium, strong>",
                    default="strong")
parser.add_argument("-o", "--output-perturbation",
                    dest="output_pert",
                    type=int,
                    help="output tolerance. invalid when robustness is strong",
                    default=1)
parser.add_argument("-a", "--activation-function",
                    dest="act_func",
                    help="activation function. choose between <none, relu, reluC, sigmoid, approx_sigmoid>",
                    default="none")

args = parser.parse_args()
input_pert = args.input_pert
robust_cons = args.robust_cons
output_pert = args.output_pert
act_func = args.act_func

print "*****Using Parameters*****"
for arg in vars(args):
  print '[' + arg + ']:', getattr(args, arg)

set_option(rational_to_decimal=True)
set_option("verbose", 10)

natureExp = RealVal(math.e)
reluC = RealVal(0.01)

print "*****Creating Weights*****"
weights1 = np.genfromtxt('mnist/para/weights1.csv', delimiter=',')
weights2 = np.genfromtxt('mnist/para/weights2.csv', delimiter=',')
weights3 = np.genfromtxt('mnist/para/weights3.csv', delimiter=',')

l0_n, l1_n = weights1.shape #748, 128
l2_n, l3_n = weights3.shape #32, 10

weights1 = np.transpose(weights1)
weights2 = np.transpose(weights2)
weights3 = np.transpose(weights3)

W1 = [ [ RealVal(weights1[i][j]) for j in range(l0_n) ] for i in range(l1_n) ]
W2 = [ [ RealVal(weights2[i][j]) for j in range(l1_n) ] for i in range(l2_n) ]
W3 = [ [ RealVal(weights3[i][j]) for j in range(l2_n) ] for i in range(l3_n) ]

print float(W1[l1_n - 1][l0_n - 1].as_decimal(20)), weights1[l1_n - 1][l0_n - 1]
print float(W2[l2_n - 1][l1_n - 1].as_decimal(20)), weights2[l2_n - 1][l1_n - 1]
print float(W3[l3_n - 1][l2_n - 1].as_decimal(20)), weights3[l3_n - 1][l2_n - 1]

print "*****Creating Biases*****"
biases1 = np.genfromtxt('mnist/para/biases1.csv', delimiter=',')
biases2 = np.genfromtxt('mnist/para/biases2.csv', delimiter=',')
biases3 = np.genfromtxt('mnist/para/biases3.csv', delimiter=',')

B1 = [ RealVal(biases1[i]) for i in range(l1_n) ]
B2 = [ RealVal(biases2[i]) for i in range(l2_n) ]
B3 = [ RealVal(biases3[i]) for i in range(l3_n) ]

print float(B1[l1_n - 1].as_decimal(20)), biases1[l1_n - 1]
print float(B2[l2_n - 1].as_decimal(20)), biases2[l2_n - 1]
print float(B3[l3_n - 1].as_decimal(20)), biases3[l3_n - 1]

print "*****Creating Assertions*****"
InX = [ Real('inX-%s' % i) for i in range(l0_n) ]
InY= [ Real('inY-%s' % i) for i in range(l0_n) ]
L1X = [ Real('l1X-%s' % i) for i in range(l1_n) ]
L1Y = [ Real('l1Y-%s' % i) for i in range(l1_n) ]
L2X = [ Real('l2X-%s' % i) for i in range(l2_n) ]
L2Y = [ Real('l2Y-%s' % i) for i in range(l2_n) ]
OutX = [ Real('outX-%s' % i) for i in range(l3_n) ]
OutY = [ Real('outY-%s' % i) for i in range(l3_n) ]

def convertToPythonNum(num):
  if is_real(num) == True:
    if is_rational_value(num) == True:
      denom = num.denominator()
      numerator = num.numerator()
      return float(numerator.as_string()) / float(denom.as_string())
    else:
      approx_num = num.approx(5)
      denom = approx_num.denominator()
      numerator = approx_num.numerator()
      return float(numerator.as_string()) / float(denom.as_string())
  else:
    return float(num.as_string())

def sigmoid(x):
  res = 1 / (1 + natureExp**(-x))
  return res

# Use the first 4 terms of the taylor series of exp
def approx_sigmoid(x):
  approxExp = 1 - x + x**2 / 2 - x**3/6
  res = 1 / (1 + approxExp)
  return res

def robust(X, Y, n):
  # robust(X, Y) is equivalent to assert argmax(X) == argmax(Y). Note that if
  # there are multiple occurrences of the max value, the standard argmax will
  # return only the first occurrence.  In that case, this |robust|
  # implementation for checking robustness is incorrect. For example, if
  # X = [4, 4, 4] and Y = [3, 5, 3], then argmax(X) is 0 and argmax(Y) is 1.
  # In this case, argmax(X) != argmax(Y) while robust(X, Y, 3) is still true
  # because And(X[1] >= X[0], X[1] >= X[2], Y[1] >= Y[0], Y[1] >= Y[2]) == True.
  return Or( [ And( [ And( X[j] >= X[i], Y[j] >= Y[i] ) for i in range(n) if i != j ] ) for j in range(n) ] )

# To Make sure all elements in X are different
def unique(X, n):
  return And([X[i] != X[j] for i in range(n) for j in range(i) if j != i])

def vvmul(V1, V2, bias, n):
  res = 0
  for i in range(0, n):
    res += V1[i] * V2[i]
  res += bias
  return res

# V=[1_row * m_col], M=[n_row * m_col]
def vmmul(V, M, B, O, m, n):
  cond = [None] * n
  for i in range(0, n):
    tmp = vvmul(V, M[i], B[i], m)
    if act_func == "relu":
      cond[i] = If(tmp > 0, O[i] == tmp, O[i] == 0)
    elif act_func == "reluC":
      cond[i] = If(tmp > 0, O[i] == tmp, O[i] == tmp * reluC)
    elif act_func == "sigmoid":
      cond[i] = (O[i] == sigmoid(tmp))
    elif act_func == "approx_sigmoid":
      cond[i] = (O[i] == approx_sigmoid(tmp))
    else:
      cond[i] = (O[i] == tmp)
  return cond

l1_x_cond = vmmul(InX, W1, B1, L1X, l0_n, l1_n)
l2_x_cond = vmmul(L1X, W2, B2, L2X, l1_n, l2_n)
out_x_cond = vmmul(L2X, W3, B3, OutX, l2_n, l3_n)

l1_y_cond = vmmul(InY, W1, B1, L1Y, l0_n, l1_n)
l2_y_cond = vmmul(L1Y, W2, B2, L2Y, l1_n, l2_n)
out_y_cond = vmmul(L2Y, W3, B3, OutY, l2_n, l3_n)

#TODO: The input pertubation constriants have to be more general
input_cond = [ And(0 < InY[i] - InX[i], InY[i] - InX[i] < input_pert, 0 < InY[i], InY[i] < 1, 0 < InX[i], InX[i] < 1) for i in range(l0_n) ]

if robust_cons == "weak":
  # This is the weakest constraint. It asserts that the new output is wrong
  # only when all labels are off by |output_pert|.
  output_cond = [ OutY[i] - OutX[i] > output_pert for i in range(l3_n) ]
elif robust_cons == "medium":
  # This is stronger than the weak constraint, but is a necessary but
  # insufficient constraint for negating robustness. It asserts an output as
  # wrong if any of the label is off by |output_pert|.
  output_cond = [ Or( [ OutY[i] - OutX[i] > output_pert for i in range(l3_n) ] ) ]
else:
  # This is the precise constraint for negating robustness (see comments for
  # |robust| implementation for details), but more complex to solve.
  output_cond = [ Not( robust(OutX, OutY, l3_n) ) ]

s = Solver()
s.add(l1_x_cond +
      l2_x_cond +
      out_x_cond +
      l1_y_cond +
      l2_y_cond +
      out_y_cond +
      input_cond +
      output_cond)
asserts = s.assertions()
print len(asserts), "constraints"

print "*****Start Solving*****"
startTime = time.time()
result = s.check()
duration = time.time() - startTime
print "[Runtime]", duration

if (result == sat):
  m = s.model()
  print m
  print "argmax(OutX)", np.argmax([convertToPythonNum(m.evaluate(OutX[i])) for i in range(l3_n)])
  print "argmax(OutY)", np.argmax([convertToPythonNum(m.evaluate(OutY[i])) for i in range(l3_n)])
else:
  print s.check()
