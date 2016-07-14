# Copyright (c) Microsoft Corporation 2015

#inputL(l0) --> outputL(l1)

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

args = parser.parse_args()
input_pert = args.input_pert
robust_cons = args.robust_cons
output_pert = args.output_pert

print "*****Using Parameters*****"
for arg in vars(args):
  print '[' + arg + ']:', getattr(args, arg)

set_option(rational_to_decimal=True)
set_option("verbose", 10)

print "\n*****Creating Weights*****"
weights = np.genfromtxt('mnist/para/softmax_weights.csv', delimiter=',')

l0_n, l1_n = weights.shape #748, 10
weights = np.transpose(weights)

# W = 10_row * 784_col
W = [ [ RealVal(weights[i][j]) for j in range(l0_n) ] for i in range(l1_n) ]

print float(W[l1_n - 1][l0_n - 1].as_decimal(20)), weights[l1_n - 1][l0_n - 1]

print "\n*****Creating Biases*****"
biases = np.genfromtxt('mnist/para/softmax_biases.csv', delimiter=',')

B = [ RealVal(biases[i]) for i in range(l1_n) ]

print float(B[l1_n - 1].as_decimal(20)), biases[l1_n - 1]

print "\n*****Creating Assertions*****"
InX = [ Real('inX-%s' % i) for i in range(l0_n) ]
InY= [ Real('inY-%s' % i) for i in range(l0_n) ]
OutX = [ Real('outX-%s' % i) for i in range(l1_n) ]
OutY = [ Real('outY-%s' % i) for i in range(l1_n) ]

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
  res = [None] * n
  for i in range(0, n):
    tmp = vvmul(V, M[i], B[i], m)
    res[i] = (O[i] == tmp)
  return res

out_x_cond = vmmul(InX, W, B, OutX, l0_n, l1_n)

out_y_cond = vmmul(InY, W, B, OutY, l0_n, l1_n)

#TODO: The input pertubation constriants have to be more general
input_cond = [ And(0 < InY[i] - InX[i], InY[i] - InX[i] < input_pert, 0 < InY[i], InY[i] < 1, 0 < InX[i], InX[i] < 1) for i in range(l0_n) ]

if robust_cons == "weak":
  # This is the weakest constraint. It asserts that the new output is wrong
  # only when all labels are off by |output_pert|.
  output_cond = [ OutY[i] - OutX[i] > output_pert for i in range(l1_n) ]
elif robust_cons == "medium":
  # This is stronger than the weak constraint, but is a necessary but
  # insufficient constraint for negating robustness. It asserts an output as
  # wrong if any of the label is off by |output_pert|.
  output_cond = [ Or( [ OutY[i] - OutX[i] > output_pert for i in range(l1_n) ] ) ]
else:
  # This is the precise constraint for negating robustness (see comments for
  # |robust| implementation for details), but more complex to solve.
  output_cond = [ Not( robust(OutX, OutY, l1_n) ) ]

s = Solver()
s.add(out_x_cond +
      out_y_cond +
      input_cond +
      output_cond)
asserts = s.assertions()
print len(asserts), "constraints"

print "\n*****Start Solving*****"
startTime = time.time()
result = s.check()
duration = time.time() - startTime
print "[Runtime]", duration

if (result == sat):
  m = s.model()
  print m
  print "argmax(OutX)", np.argmax([convertToPythonNum(m.evaluate(OutX[i])) for i in range(l1_n)])
  print "argmax(OutY)", np.argmax([convertToPythonNum(m.evaluate(OutY[i])) for i in range(l1_n)])
else:
  print s.check()
