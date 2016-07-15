#inputL(l0) --> outputL(l1)

from z3 import *
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Verdict harness.")
parser.add_argument("-i", "--input-perturbation",
                    dest="input_bound",
                    type=float,
                    help="input perturbation. choose between (0, 1)",
                    default=0.0001)
parser.add_argument("-r", "--robustness",
                    dest="robust_cons",
                    help="robustness constraint. choose between <imprecise, precise>",
                    default="precise")
parser.add_argument("-o", "--output-bound",
                    dest="output_bound",
                    type=int,
                    help="output bound. invalid when robustness is strong",
                    default=1)
parser.add_argument("-m", "--verify-mode",
                    dest="verify_mode",
                    help="verification mode. choose between <general, specific>",
                    default="general")

args = parser.parse_args()
input_bound = args.input_bound
robust_cons = args.robust_cons
output_bound = args.output_bound
verify_mode = args.verify_mode

print "\n=====Parameter List====="
for arg in vars(args):
  print '[' + arg + ']:', getattr(args, arg)

set_option(rational_to_decimal=True)
set_option("verbose", 10)

def convertToPythonNum(num):
  if is_real(num) == True:
    if is_rational_value(num) == True:
      # 1 doesn't end with '?'
      if num.as_decimal(10).endswith('?'):
        return float(num.as_decimal(10)[:-1])
      else:
        return float(num.as_decimal(10))
    else:
      # |approx_num| is the approx rational of the irrational |num|
      # |approx_num| is guaranteed to end with a '?'
      approx_num = num.approx(5)
      return float(approx_num.as_decimal(10)[:-1])
  else:
    return float(num.as_string())

def robust(X, Y, n):
  if robust_cons == "imprecise":
    # This is an imprecise constraint. It asserts that a model is robust if
    # *all* labels' errors are bound by |output_bound|. Depending on the value
    # of |output_bound|, this constraint could be more relaxed or stricter than
    # the precise constraint. Relaxed or strict, it is easier to solve.
    return And( [ And(Y[i] - X[i] < output_bound, X[i] - Y[i] < output_bound) for i in range(n) ] )
  else:
    # This is the precise constraint for robustness, but more complex to solve.
    # It is equivalent to assert argmax(X) == argmax(Y). Note that if there are
    # multiple occurrences of the max value, the standard argmax will return
    # only the first occurrence.  In that case, this |robust| implementation
    # for checking robustness is incorrect. For example, if X = [4, 4, 4] and Y
    # = [3, 5, 3], then argmax(X) is 0 and argmax(Y) is 1.  In this case,
    # argmax(X) != argmax(Y) while robust(X, Y, 3) is still true because
    # And(X[1] >= X[0], X[1] >= X[2], Y[1] >= Y[0], Y[1] >= Y[2]) == True.
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

def solveIt(s):
  #print "=====Start Solving====="
  startTime = time.time()
  result = s.check()
  duration = time.time() - startTime
  print "[Solver Runtime] %.2f %s" % (duration, result)

  if (result == sat):
    m = s.model()
    #print m
    print "argmax(OutX)", np.argmax([convertToPythonNum(m.evaluate(OutX[i])) for i in range(l1_n)])
    print "argmax(OutY)", np.argmax([convertToPythonNum(m.evaluate(OutY[i])) for i in range(l1_n)])

print "\nCreating Weights"
weights = np.genfromtxt('mnist/para/softmax_weights.csv', delimiter=',')
l0_n, l1_n = weights.shape #748, 10
weights = np.transpose(weights)
# W = 10_row * 784_col
W = [ [ RealVal(weights[i][j]) for j in range(l0_n) ] for i in range(l1_n) ]

print "Creating Biases"
biases = np.genfromtxt('mnist/para/softmax_biases.csv', delimiter=',')
B = [ RealVal(biases[i]) for i in range(l1_n) ]

print "Creating Assertions"
if verify_mode == "specific":
  # The MNIST data set has 10,000 images for testing
  inputs = np.genfromtxt('mnist/para/mnist_test_images_100.csv', delimiter=',')
  InX = [ RealVal(inputs[0][i]) for i in range(l0_n) ]
else:
  InX= [ Real('inX-%s' % i) for i in range(l0_n) ]
InY= [ Real('inY-%s' % i) for i in range(l0_n) ]
OutX = [ Real('outX-%s' % i) for i in range(l1_n) ]
OutY = [ Real('outY-%s' % i) for i in range(l1_n) ]

out_x_cond = vmmul(InX, W, B, OutX, l0_n, l1_n)
out_y_cond = vmmul(InY, W, B, OutY, l0_n, l1_n)
input_cond = [ And(InX[i] - InY[i] < input_bound, InY[i] - InX[i] < input_bound, 0 <= InY[i], InY[i] <= 1, 0 <= InX[i], InX[i] <= 1) for i in range(l0_n) ]
output_cond = [ Not( robust(OutX, OutY, l1_n) ) ]

s = Solver()
s.add(out_x_cond +
      out_y_cond+
      input_cond +
      output_cond)
solveIt(s)
