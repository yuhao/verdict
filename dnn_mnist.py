#inputL(l0) --> hiddenL1(l1) --> hiddenL2(l2) --> outputL(l3)

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
parser.add_argument("-a", "--activation-function",
                    dest="act_func",
                    help="activation function. choose between <none, relu, reluC, sigmoid, approx_sigmoid>",
                    default="none")
parser.add_argument("-m", "--verify-mode",
                    dest="verify_mode",
                    help="verification mode. choose between <general, specific>",
                    default="general")

args = parser.parse_args()
input_bound = args.input_bound
robust_cons = args.robust_cons
output_bound = args.output_bound
act_func = args.act_func
verify_mode = args.verify_mode

print "\n=====Parameter List====="
for arg in vars(args):
  print '[' + arg + ']:', getattr(args, arg)

set_option(rational_to_decimal=True)
set_option("verbose", 10)

natureExp = RealVal(math.e)
reluC = RealVal(0.01)

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

def sigmoid(x):
  res = 1 / (1 + natureExp**(-x))
  return res

# Use the first 4 terms of the taylor series of exp
def approx_sigmoid(x):
  approxExp = 1 - x + x**2 / 2 - x**3/6
  res = 1 / (1 + approxExp)
  return res

def vvmul(V1, V2, bias, n):
  res = 0
  for i in range(0, n):
    res += V1[i] * V2[i]
  res += bias
  return res

# V=[1_row * m_col], M=[n_row * m_col]
def vmmul(V, M, B, m, n):
  res = [None] * n
  for i in range(0, n):
    tmp = vvmul(V, M[i], B[i], m)
    if act_func == "relu":
      res[i] = If(tmp > 0, tmp, 0)
    elif act_func == "reluC":
      res[i] = If(tmp > 0, tmp, tmp * reluC)
    elif act_func == "sigmoid":
      res[i] = sigmoid(tmp)
    elif act_func == "approx_sigmoid":
      res[i] = approx_sigmoid(tmp)
    else:
      res[i] = tmp
  return res

def solveIt(n):
  startTime = time.time()
  result = s.check()
  duration = time.time() - startTime
  print "[Solver Runtime] %.2f %s" % (duration, result)

  if (result == sat):
    m = s.model()
    #print m
    print "argmax(OutX)", np.argmax([convertToPythonNum(m.evaluate(OutX[i])) for i in range(n)])
    print "argmax(OutY)", np.argmax([convertToPythonNum(m.evaluate(OutY[i])) for i in range(n)])

print "\nCreating Weights"
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

print "Creating Biases"
biases1 = np.genfromtxt('mnist/para/biases1.csv', delimiter=',')
biases2 = np.genfromtxt('mnist/para/biases2.csv', delimiter=',')
biases3 = np.genfromtxt('mnist/para/biases3.csv', delimiter=',')

B1 = [ RealVal(biases1[i]) for i in range(l1_n) ]
B2 = [ RealVal(biases2[i]) for i in range(l2_n) ]
B3 = [ RealVal(biases3[i]) for i in range(l3_n) ]

print "Creating Assertions"
if verify_mode == "specific":
  inputs = np.genfromtxt('mnist/para/mnist_test_images_100.csv', delimiter=',')

  # The MNIST data set has 10,000 images for testing
  #num_imgs = 100
  #InX = [ [ RealVal(inputs[j][i]) for i in range(l0_n) ] for j in range(num_imgs) ]
  InX = [ RealVal(inputs[0][i]) for i in range(l0_n) ]
else:
  InX = [ Real('inX-%s' % i) for i in range(l0_n) ]
InY= [ Real('inY-%s' % i) for i in range(l0_n) ]

L1X = vmmul(InX, W1, B1, l0_n, l1_n)
L2X = vmmul(L1X, W2, B2, l1_n, l2_n)
OutX = vmmul(L2X, W3, B3, l2_n, l3_n)

L1Y = vmmul(InY, W1, B1, l0_n, l1_n)
L2Y = vmmul(L1Y, W2, B2, l1_n, l2_n)
OutY = vmmul(L2Y, W3, B3, l2_n, l3_n)

#TODO: The input pertubation constriants have to be more general
input_cond = [ And(InX[i] - InY[i] < input_bound, InY[i] - InX[i] < input_bound, 0 <= InY[i], InY[i] <= 1, 0 <= InX[i], InX[i] <= 1) for i in range(l0_n) ]

output_cond = [ Not( robust(OutX, OutY, l3_n) ) ]

s = Solver()
s.add(input_cond +
      output_cond)
solveIt(l3_n)

