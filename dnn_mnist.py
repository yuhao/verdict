#inputL(l0) --> hiddenL1(l1) --> hiddenL2(l2) --> outputL(l3)

from z3 import *
import numpy as np
import argparse
import verdict_core as core

parser = argparse.ArgumentParser(description="Verdict harness.")
parser.add_argument("-i", "--input-perturbation",
                    dest="input_bound",
                    type=float,
                    help="input perturbation. choose between (0, 1)",
                    default=0.004)
parser.add_argument("-r", "--robustness",
                    dest="robust_cons",
                    help="robustness constraint. choose between <imprecise, precise>",
                    default="precise")
parser.add_argument("-o", "--output-bound",
                    dest="output_bound",
                    type=int,
                    help="output bound. invalid when robustness is precise",
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
  # The MNIST data set has 10,000 images for testing
  inputs = np.genfromtxt('mnist/para/mnist_test_images_100.csv', delimiter=',')
  InX = [ RealVal(inputs[0][i]) for i in range(l0_n) ]
else:
  InX = [ Real('inX-%s' % i) for i in range(l0_n) ]
InY= [ Real('inY-%s' % i) for i in range(l0_n) ]

L1X = core.vmmul(InX, W1, B1, l0_n, l1_n, act_func)
L2X = core.vmmul(L1X, W2, B2, l1_n, l2_n, act_func)
OutX = core.vmmul(L2X, W3, B3, l2_n, l3_n, act_func)

L1Y = core.vmmul(InY, W1, B1, l0_n, l1_n, act_func)
L2Y = core.vmmul(L1Y, W2, B2, l1_n, l2_n, act_func)
OutY = core.vmmul(L2Y, W3, B3, l2_n, l3_n, act_func)

#TODO: The input pertubation constriants have to be more general
input_cond = [ And(InX[i] - InY[i] < input_bound, InY[i] - InX[i] < input_bound, 0 <= InY[i], InY[i] <= 1, 0 <= InX[i], InX[i] <= 1) for i in range(l0_n) ]

output_cond = [ Not( core.full_robust(OutX, OutY, l3_n, robust_cons, bound=output_bound) ) ]

s = Solver()
s.add(input_cond +
      output_cond)
result = core.solveIt(s)
if (result == sat):
  m = s.model()
  #print m
  outx = [m.evaluate(OutX[i]) for i in range(l3_n)]
  outy = [m.evaluate(OutY[i]) for i in range(l3_n)]
  print "OutX", outx
  print "OutY", outy
  print "argmax(OutX)", core.argmax(outx)
  print "argmax(OutY)", core.argmax(outy)
