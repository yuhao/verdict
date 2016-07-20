#inputL(l0) --> outputL(l1)

from z3 import *
import numpy as np
import argparse
import verdict_core as core

parser = argparse.ArgumentParser(description="Verdict harness.")
parser.add_argument("-i", "--input-info",
                    dest="input_info",
                    nargs=2,
                    help="input info. a tuple of [scaled/unscaled, perturbation]. choose perturbation between [0, 255] if unscaled or [0, 1] if scaled)",
                    default=['scaled', '0.004'])
parser.add_argument("-r", "--robustness",
                    dest="robust_cons",
                    help="robustness constraint. choose between <imprecise, precise>",
                    default="precise")
parser.add_argument("-o", "--output-bound",
                    dest="output_bound",
                    type=int,
                    help="output bound. invalid when robustness is precise",
                    default=1)
parser.add_argument("-m", "--verify-mode",
                    dest="verify_mode",
                    help="verification mode. choose between <general, specific>",
                    default="general")
parser.add_argument("-c", "--input-id",
                    dest="input_id",
                    type=int,
                    help="input id. used in the specific mode",
                    default=0)

args = parser.parse_args()
input_info = args.input_info
input_scaled = input_info[0]
input_var = float(input_info[1]) if input_scaled == "scaled" else int(input_info[1])
robust_cons = args.robust_cons
output_bound = args.output_bound
verify_mode = args.verify_mode
input_id = args.input_id

print "\n=====Parameter List====="
for arg in vars(args):
  print '[' + arg + ']:', getattr(args, arg)

set_option(rational_to_decimal=True)
set_option("verbose", 10)

def genCExp(m):
  with file('cexp.csv', 'w') as outfile:
    inx = [convertToPythonNum(m.evaluate(X[i])) for i in range(l0_n)]
    np.savetxt(outfile, np.atleast_2d(inx), delimiter=",")
    iny = [convertToPythonNum(m.evaluate(Y[i])) for i in range(l0_n)]
    np.savetxt(outfile, np.atleast_2d(iny), delimiter=",")

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
if (input_scaled == "scaled"):
  if verify_mode == "specific":
    inputs = np.genfromtxt('mnist/para/mnist_test_images_100.csv', delimiter=',')
    InX = [ RealVal(inputs[0][i]) for i in range(l0_n) ]
  else:
    InX = [ Real('inX-%s' % i) for i in range(l0_n) ]
  InY= [ Real('inY-%s' % i) for i in range(l0_n) ]
else: #unscaled
  scale = RealVal(255)
  if verify_mode == "specific":
    inputs = np.genfromtxt('mnist/para/mnist_test_images_unscaled_100.csv', delimiter=',')
    X = [ IntVal(inputs[input_id][i]) for i in range(l0_n) ]
    InX = core.scaleInput(X, scale, l0_n)
  else:
    X = [ Int('x%s' % i) for i in range(l0_n) ]
    InX = core.scaleInput(X, scale, l0_n)
  Y = [ Int('y%s' % i) for i in range(l0_n) ]
  InY = core.scaleInput(Y, scale, l0_n)

OutX = core.vmmul(InX, W, B, l0_n, l1_n, "none")
OutY = core.vmmul(InY, W, B, l0_n, l1_n, "none")
if input_scaled == "scaled":
  input_cond = [ And(InX[i] - InY[i] < input_var, InY[i] - InX[i] < input_var, 0 <= InY[i], InY[i] <= 1, 0 <= InX[i], InX[i] <= 1) for i in range(l0_n) ]
else:
  input_cond = [ And(X[i] - Y[i] < input_var, Y[i] - X[i] < input_var, 0 <= Y[i], Y[i] <= 255, 0 <= X[i], X[i] <= 255) for i in range(l0_n) ]
output_cond = [ Not( core.full_robust(OutX, OutY, l1_n, robust_cons) ) ]

s = Solver()
s.add(input_cond +
      output_cond)
result = core.solveIt(s)
if (result == sat):
  m = s.model()
  #print m
  outx = [m.evaluate(OutX[i]) for i in range(l1_n)]
  outy = [m.evaluate(OutY[i]) for i in range(l1_n)]
  print "OutX", outx
  print "OutY", outy
  print "argmax(OutX)", np.argmax(outx)
  print "argmax(OutY)", np.argmax(outy)
  genCExp(m)
