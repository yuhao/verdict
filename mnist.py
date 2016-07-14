# Copyright (c) Microsoft Corporation 2015

#inputL(l0) --> hiddenL1(l1) --> hiddenL2(l2) --> outputL(l3)

from z3 import *
import numpy as np
import time

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

W1 = [ [ RealVal(weights1[i][j]) for j in range(l1_n) ] for i in range(l0_n) ]
W2 = [ [ RealVal(weights2[i][j]) for j in range(l2_n) ] for i in range(l1_n) ]
W3 = [ [ RealVal(weights3[i][j]) for j in range(l3_n) ] for i in range(l2_n) ]

print float(W1[l0_n - 1][l1_n - 1].as_decimal(20)), weights1[l0_n - 1][l1_n - 1]
print float(W2[l1_n - 1][l2_n - 1].as_decimal(20)), weights2[l1_n - 1][l2_n - 1]
print float(W3[l2_n - 1][l3_n - 1].as_decimal(20)), weights3[l2_n - 1][l3_n - 1]

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

def sigmoid(x):
  res = 1 / (1 + natureExp**(-x))
  return res

# Use the first 4 terms of the taylor series of exp
def approx_sigmoid(x):
  approxExp = 1 - x + x**2 / 2 - x**3/6
  res = 1 / (1 + approxExp)
  return res

def robust(X, Y, n):
  # robust(X, Y) is equivalent to assert argmax(X) == argmax(Y)
  return Or( [ And( [ And( X[j] >= X[i], Y[j] >= Y[i] ) for i in range(n) if i != j ] ) for j in range(n) ] )

def unique(X, n):
  return And([X[i] != X[j] for i in range(n) for j in range(i) if j != i])

def vvmul(V1, V2, n):
  res = 0
  for i in range(0, n):
    res += V1[i] * V2[i]
  return res

def vmmul(V, M, O, n):
  res = [None] * n
  for i in range(0, n):
    tmp = vvmul(M[i], V, n)
    ### Use ReLU for activatation
    res[i] = (O[i] == tmp)
    #res[i] = If(tmp > 0, O[i] == tmp, O[i] == 0)
    #res[i] = If(tmp >= 0, O[i] == tmp, O[i] == tmp * reluC)
    ### Use sigmoid for activatation
    #res[i] = (O[i] == sigmoid(tmp))
    #res[i] = (O[i] == approx_sigmoid(tmp))
  return res

l1_x_cond = vmmul(InX, W1, L1X, l1_n)
l2_x_cond = vmmul(L1X, W2, L2X, l2_n)
out_x_cond = vmmul(L2X, W3, OutX, l3_n)

l1_y_cond = vmmul(InY, W1, L1Y, l1_n)
l2_y_cond = vmmul(L1Y, W2, L2Y, l2_n)
out_y_cond = vmmul(L2Y, W3, OutY, l3_n)

input_cond = [ And(0 < InY[i] - InX[i], InY[i] - InX[i] < 0.0001, 0 <= InY[i], InY[i] <= 1, 0 <= InX[i], InX[i] <= 1) for i in range(l0_n) ]

# This is a necessary but not sufficient constraint for negating robustness (for classification)
output_cond = [ Or( [ OutY[i] - OutX[i] > 1 for i in range(l3_n) ] ) ]
# This is a precise constraint for negating robustness, but more complex to solve
#output_cond = [ Not(robust(OutX, OutY, l3_n)) ]

s = Solver()
s.add(l1_x_cond +
      l2_x_cond +
      out_x_cond +
      l1_y_cond +
      l2_y_cond +
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
  print "argmax(OutX)", np.argmax([float(m.evaluate(OX[i]).as_decimal(20)) for i in range(5)])
  print "argmax(OutY)", np.argmax([float(m.evaluate(OY[i]).as_decimal(20)) for i in range(5)])
else:
  print s.check()
print "[Runtime]", duration
