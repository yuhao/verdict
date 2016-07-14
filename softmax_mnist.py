# Copyright (c) Microsoft Corporation 2015

#inputL(l0) --> outputL(l1)

from z3 import *
import numpy as np
import time

set_option(rational_to_decimal=True)
set_option("verbose", 10)

natureExp = RealVal(math.e)
reluC = RealVal(0.01)

print "*****Creating Weights*****"
weights = np.genfromtxt('mnist/para/softmax_weights.csv', delimiter=',')

l0_n, l1_n = weights.shape #748, 10
weights = np.transpose(weights)

# W = 10_row * 784_col
W = [ [ RealVal(weights[i][j]) for j in range(l0_n) ] for i in range(l1_n) ]

print float(W[l1_n - 1][l0_n - 1].as_decimal(20)), weights[l1_n - 1][l0_n - 1]

print "*****Creating Biases*****"
biases = np.genfromtxt('mnist/para/softmax_biases.csv', delimiter=',')

B = [ RealVal(biases[i]) for i in range(l1_n) ]

print float(B[l1_n - 1].as_decimal(20)), biases[l1_n - 1]

print "*****Creating Assertions*****"
InX = [ Real('inX-%s' % i) for i in range(l0_n) ]
InY= [ Real('inY-%s' % i) for i in range(l0_n) ]
OutX = [ Real('outX-%s' % i) for i in range(l1_n) ]
OutY = [ Real('outY-%s' % i) for i in range(l1_n) ]

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
input_cond = [ And(0 < InY[i] - InX[i], InY[i] - InX[i] < 0.0001, 0 < InY[i], InY[i] < 1, 0 < InX[i], InX[i] < 1) for i in range(l0_n) ]

# This is a necessary but not sufficient constraint for negating robustness (for classification)
#output_cond = [ Or( [ OutY[i] - OutX[i] > 1 for i in range(l1_n) ] ) ]
# This is a precise constraint for negating robustness, but more complex to solve
output_cond = [ Not( robust(OutX, OutY, l1_n) ) ]

s = Solver()
s.add(out_x_cond +
      out_y_cond +
      input_cond +
      output_cond)
#asserts = s.assertions()
#print len(asserts), "constraints"

print "*****Start Solving*****"
startTime = time.time()
result = s.check()
duration = time.time() - startTime

if (result == sat):
  m = s.model()
  print m
  print "argmax(OutX)", np.argmax([float(m.evaluate(OutX[i]).as_decimal(20)) for i in range(l1_n)])
  print "argmax(OutY)", np.argmax([float(m.evaluate(OutY[i]).as_decimal(20)) for i in range(l1_n)])
else:
  print s.check()
print "[Runtime]", duration
