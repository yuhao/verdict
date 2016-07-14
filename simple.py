# Copyright (c) Microsoft Corporation 2015

from z3 import *
import sys
import numpy as np

X = [ Int('x%s' % i) for i in range(5) ]
Y = [ Int('y%s' % i) for i in range(5) ]
L1X = [ Real('l1-x-%s' % i) for i in range(5) ]
L1Y = [ Real('l1-y-%s' % i) for i in range(5) ]
OX = [ Real('ox%s' % i) for i in range(5) ]
OY = [ Real('oy%s' % i) for i in range(5) ]
###This weight matrix leads to sat, i.e., the model is not robust
W1 = [[RealVal(0.150231045), RealVal(0.670946632), RealVal(0.038289158), RealVal(0.673947442), RealVal(0.778584339)],
      [RealVal(0.609666936), RealVal(0.829361313), RealVal(0.769045319), RealVal(0.090266416), RealVal(0.611124522)],
      [RealVal(0.372291187), RealVal(0.296078870), RealVal(0.071903055), RealVal(0.329610558), RealVal(0.497106420)],
      [RealVal(0.956393326), RealVal(0.798037186), RealVal(0.286287309), RealVal(0.674712053), RealVal(0.840121924)],
      [RealVal(0.627576681), RealVal(0.975366059), RealVal(0.809793115), RealVal(0.538486285), RealVal(0.508208182)]]
###This weight matrix leads to unsat, i.e., the model is robust
#W1 = [[RealVal(0.898798758), RealVal(0.737760202), RealVal(-0.340628972), RealVal(0.359962019), RealVal(-0.653788730)],
#      [RealVal(0.286695467), RealVal(0.636300932), RealVal(0.308495501), RealVal(-0.280221795), RealVal(0.227396977)],
#      [RealVal(0.347919571), RealVal(-0.014086935), RealVal(-0.890731476), RealVal(0.319317432), RealVal(0.445368357)],
#      [RealVal(-0.664655216), RealVal(-0.902996018), RealVal(-0.062735416), RealVal(-0.328387565), RealVal(-0.455935177)],
#      [RealVal(0.787489188), RealVal(0.729626432), RealVal(-0.762977967), RealVal(0.393125952), RealVal(-0.250256052)]]
W2 = [[RealVal(0.296015539), RealVal(0.298796651), RealVal(0.218944852), RealVal(0.194505800), RealVal(0.737771838)],
      [RealVal(0.333778784), RealVal(0.905985998), RealVal(0.673450123), RealVal(0.823922821), RealVal(0.049145588)],
      [RealVal(0.049989531), RealVal(0.189385222), RealVal(0.613595088), RealVal(0.763298634), RealVal(0.432115047)],
      [RealVal(0.880362946), RealVal(0.432980229), RealVal(0.070170702), RealVal(0.707604930), RealVal(0.591374432)],
      [RealVal(0.202527461), RealVal(0.073010964), RealVal(0.534760797), RealVal(0.478653934), RealVal(0.476729378)]]
valueW1 = [[0.150231045, 0.670946632, 0.038289158, 0.673947442, 0.778584339],
           [0.609666936, 0.829361313, 0.769045319, 0.090266416, 0.611124522],
           [0.372291187, 0.296078870, 0.071903055, 0.329610558, 0.497106420],
           [0.956393326, 0.798037186, 0.286287309, 0.674712053, 0.840121924],
           [0.627576681, 0.975366059, 0.809793115, 0.538486285, 0.508208182]]
valueW2 = [[0.296015539, 0.298796651, 0.218944852, 0.194505800, 0.737771838],
           [0.333778784, 0.905985998, 0.673450123, 0.823922821, 0.049145588],
           [0.049989531, 0.189385222, 0.613595088, 0.763298634, 0.432115047],
           [0.880362946, 0.432980229, 0.070170702, 0.707604930, 0.591374432],
           [0.202527461, 0.073010964, 0.534760797, 0.478653934, 0.476729378]]
natureExp = RealVal(math.e)
reluC = RealVal(0.01)

set_option(rational_to_decimal=True)
set_option("verbose", 20)

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

def sanity(v, w1, w2, n):
  hl = [0] * n
  out = [0] * n
  for i in range(n):
    for j in range(n):
      hl[i] += v[j] * w1[i][j]
      if (hl[i] < 0):
        hl[i] = 0
  for i in range(n):
    for j in range(n):
      out[i] += hl[j] * w2[i][j]
      if (out[i] < 0):
        out[i] = 0
  return out

def sanityCheck(X, Y, m, n):
  valueX = [0] * n
  for i in range(n):
    valueX[i] = m.evaluate(X[i]).as_long()
  valueY = [0] * n
  for i in range(n):
    valueY[i] = m.evaluate(Y[i]).as_long()

  sanityOutX = sanity(valueX, valueW1, valueW2, n)
  sanityOutY = sanity(valueY, valueW1, valueW2, n)
  print "Sanity-Out-X", sanityOutX
  print "Sanity-Out-Y", sanityOutY

def sigmoid(x):
  res = 1 / (1 + natureExp**(-x))
  return res

# Use the first 4 terms of the taylor series of exp
def approx_sigmoid(x):
  approxExp = 1 - x + x**2 / 2 - x**3/6
  res = 1 / (1 + approxExp)
  return res

def robust(X, Y, n):
  return Or( [ And( [ And( X[j] > X[i], Y[j] > Y[i] ) for i in range(n) if i != j ] ) for j in range(n) ] )

def vvmul(V1, V2, n):
  res = 0
  for i in range(0, n):
    res += V1[i] * V2[i]
  return res

def vmmul(V, M, O, n):
  res = [None] * n
  for i in range(0, n):
    ### Use ReLU for activatation
    tmp = vvmul(M[i], V, n)
    res[i] = (O[i] == If(tmp > 0, tmp, 0))
    #res[i] = If(tmp > 0, O[i] == tmp, O[i] == 0)
    #res[i] = If(tmp >= 0, O[i] == tmp, O[i] == tmp * reluC)
    ### Use sigmoid for activatation
    #res[i] = (O[i] == approx_sigmoid(tmp))
  return res

l1_x_cond = vmmul(X, W1, L1X, 5)
out_x_cond = vmmul(L1X, W2, OX, 5)

l1_y_cond = vmmul(Y, W1, L1Y, 5)
out_y_cond = vmmul(L1Y, W2, OY, 5)

input_cond = [ And(0 < Y[i] - X[i], Y[i] - X[i] < 2, 0 <= Y[i], Y[i] < 32, 0 <= X[i], X[i] < 32) for i in range(5) ]

output_cond = [ Not(robust(OX, OY, 5)) ]

#s = Then('simplify', 'nlsat').solver()
s = Solver()
s.add(l1_x_cond +
      out_x_cond +
      l1_y_cond +
      out_y_cond +
      input_cond +
      output_cond)
#for c in s.assertions():
#  print c
#with open("cons.out", "w") as cons_out:
#  for c in s.assertions():
#    cons_out.write(str(c)+"\n")

if (s.check() == sat):
  m = s.model()
  print "X", [m.evaluate(X[i]) for i in range(5)]
  print "Y", [m.evaluate(Y[i]) for i in range(5)]
  print "L1-X", [m.evaluate(L1X[i]) for i in range(5)]
  print "L1-Y", [m.evaluate(L1Y[i]) for i in range(5)]
  print "Out-X", [float(m.evaluate(OX[i]).as_decimal(20)) for i in range(5)]
  print "Out-Y", [float(m.evaluate(OY[i]).as_decimal(20)) for i in range(5)]
  print "argmax(Out-X)", np.argmax([convertToPythonNum(m.evaluate(OX[i])) for i in range(5)])
  print "argmax(Out-Y)", np.argmax([convertToPythonNum(m.evaluate(OY[i])) for i in range(5)])
  sanityCheck(X, Y, m, 5)
else:
  print s.check()
