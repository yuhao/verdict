#inputL(l0) --> hiddenL1(l1) --> hiddenL2(l2) --> outputL(l3)

from z3 import *
import numpy as np
import time
import argparse
import verdict_core as core

parser = argparse.ArgumentParser(description="Verdict harness.")
parser.add_argument("-i", "--input-perturbation",
                    dest="input_var",
                    type=float,
                    help="input perturbation. choose between (0, 1)",
                    default=0.004)
parser.add_argument("-c", "--input-id",
                    dest="input_id",
                    type=int,
                    help="input id. used in the specific mode",
                    default=0)

args = parser.parse_args()
input_var = args.input_var
input_id = args.input_id

print "\n=====Parameter List====="
for arg in vars(args):
  print '[' + arg + ']:', getattr(args, arg)

set_option(rational_to_decimal=True)
set_option("verbose", 10)

# V=[1_row * m_col], M=[n_row * m_col]
def vmmul(V1, V2, M, B, m, n, act):
  res1 = [None] * n
  res2 = [None] * n
  cond = [None] * n
  for i in range(0, n):
    tmp1 = core.vvmul(V1, M[i], B[i], m)
    tmp2 = core.vvmul(V2, M[i], B[i], m)

    if act == True:
      if (simplify(tmp1 > 0)):
        res1[i] = tmp1
        cond[i] = (tmp2 > 0)
        res2[i] = tmp2
      else:
        res1[i] = 0
        cond[i] = (tmp2 <= 0)
        res2[i] = 0
    else:
      res1[i] = tmp1
      res2[i] = tmp2
  return (res1, res2, cond)

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
inputs = np.genfromtxt('mnist/para/mnist_test_images_100.csv', delimiter=',')
InX = [ RealVal(inputs[input_id][i]) for i in range(l0_n) ]
InY= [ Real('inY-%s' % i) for i in range(l0_n) ]

L1X, L1Y, cond1 = vmmul(InX, InY, W1, B1, l0_n, l1_n, True)
L2X, L2Y, cond2 = vmmul(L1X, L1Y, W2, B2, l1_n, l2_n, True)
OutX, OutY, _ = vmmul(L2X, L2Y, W3, B3, l2_n, l3_n, False)

#TODO: The input pertubation constriants have to be more general
input_cond = [ And(InX[i] - InY[i] < input_var, InY[i] - InX[i] < input_var, 0 <= InY[i], InY[i] <= 1, 0 <= InX[i], InX[i] <= 1) for i in range(l0_n) ]

robust_cond = core.ref_robust(OutX, OutY, l3_n)
output_cond = [ Not(robust_cond[i]) for i in range(l3_n - 1) ]

s = Solver()
s.add(input_cond)
result = core.solveIt(s)
print "[add] %d asts now" % (len(s.assertions()))

# Incrementally adding constraints. The complete constraint takes the
# following form: IL /\ HL1 /\ HL2 /\ (OL1 \/ OL2 ... \/ OL9), where IL is
# input_cond, HL1 and HL2 are cond1 and cond2, and OLn is individual formulas in
# output_cond. The idea is to check IL /\ OL1 first, if unsat, try IL /\ OL2,
# and so on. Best case we can quickly get unsat and verifies robustness. If
# IL /\ OL1 is sat, then incrementally add formulas in HL1 and HL2. If unsat,
# move to next OL, if sat, keep adding them.
for index, out_cond in enumerate(output_cond):
  print "Adding constraint %d from output layer" % index
  s.push()
  s.add(out_cond)
  print "[push] %d asts now" % (len(s.assertions()))
  result = core.solveIt(s)
  if (result == unsat):
    s.pop()
    print "[pop] %d asts now" % (len(s.assertions()))
    continue
  else:
    s.push()
    for index, cond in enumerate(cond1 + cond2):
      if (index < l1_n):
        print "Adding constraint %d from hidden layer 1" % index
      else:
        print "Adding constraint %d from hidden layer 2" % index
      s.add(cond)
      print "[push] %d asts now" % (len(s.assertions()))
      result = core.solveIt(s)
      if (result == unsat):
        break
    s.pop()
    print "[pop] %d asts now" % (len(s.assertions()))
    s.pop()
    print "[pop] %d asts now" % (len(s.assertions()))
    if (result == sat):
      break;

if (result == unsat):
  print "Model is robust on image %d against perturbation %f!" % (input_id, input_var)
else:
  m = s.model()
  outx = [simplify(OutX[i]) for i in range(l3_n)]
  outy = [m.evaluate(OutY[i]) for i in range(l3_n)]
  print "OutX", outx
  print "OutY", outy
  print "argmax(OutX)", np.argmax(outx)
  print "argmax(OutY)", np.argmax(outy)
