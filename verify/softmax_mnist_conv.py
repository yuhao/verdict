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
parser.add_argument("-c", "--input-id",
                    dest="input_id",
                    type=int,
                    help="input id. used in the specific mode",
                    default=0)

args = parser.parse_args()
input_info = args.input_info
input_scaled = input_info[0]
input_var = float(input_info[1]) if input_scaled == "scaled" else int(input_info[1])
output_bound = args.output_bound
input_id = args.input_id

print "\n=====Parameter List====="
for arg in vars(args):
  print '[' + arg + ']:', getattr(args, arg)

set_option(rational_to_decimal=True)
set_option("verbose", 10)

print "\nCreating Weights"
weights = np.genfromtxt('../mnist/para/softmax_weights.csv', delimiter=',')
l0_n, l1_n = weights.shape #748, 10
weights = np.transpose(weights)
# W = 10_row * 784_col
W = [ [ RealVal(weights[i][j]) for j in range(l0_n) ] for i in range(l1_n) ]

print "Creating Biases"
biases = np.genfromtxt('../mnist/para/softmax_biases.csv', delimiter=',')
B = [ RealVal(biases[i]) for i in range(l1_n) ]

print "Creating Assertions"
InX = [ Real('inX-%s' % i) for i in range(l0_n) ]
kernel = core.gaussian_kernel(2, 1)
InY = core.convolve(InX, kernel, int(math.sqrt(l0_n)))

# Same as full_robust except it returns the CNF form (i.e., without ORing them)
def robustCNF(X, Y, n):
  return [ And( [ And( X[j] >= X[i], Y[j] >= Y[i] ) for i in range(n) if i != j ] ) for j in range(n) ]

OutX = core.vmmul(InX, W, B, l0_n, l1_n, "none")
OutY = core.vmmul(InY, W, B, l0_n, l1_n, "none")
input_cond = [ And(0 <= InY[i], InY[i] <= 1, 0 <= InX[i], InX[i] <= 1) for i in range(l0_n) ]
robust_cond = robustCNF(OutX, OutY, l1_n) # OR these together
output_cond = [ Not(robust_cond[i]) for i in range(l1_n) ]

s = Solver()
s.add(input_cond)
#print "[add] %d asts now" % (len(s.assertions()))
result = core.solveIt(s)

for index, out_cond in enumerate(output_cond):
  print "Adding constraint %d from output layer" % index
  s.add(out_cond)
  #print "[add] %d asts now" % (len(s.assertions()))
  result = core.solveIt(s)
  if (result == unsat):
    break;

if (result == sat):
  m = s.model()
  outx = [m.evaluate(OutX[i]) for i in range(l1_n)]
  outy = [m.evaluate(OutY[i]) for i in range(l1_n)]
  print "OutX", outx
  print "OutY", outy
  print "argmax(OutX)", core.argmax(outx)
  print "argmax(OutY)", core.argmax(outy)
  core.genCounterExp('/tmp/softmax_mnist_conv.csv', m, InX, InY)
