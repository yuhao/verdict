# Copyright (c) Microsoft Corporation 2015

from z3 import *

X = [ Real('x%s' % i) for i in range(5) ]
Y = [ Real('y%s' % i) for i in range(5) ]
L1X = [ Real('l1-x-%s' % i) for i in range(5) ]
L1Y = [ Real('l1-y-%s' % i) for i in range(5) ]
OX = [ Real('ox%s' % i) for i in range(5) ]
OY = [ Real('oy%s' % i) for i in range(5) ]
W1 = [ [ RealVal(2.2), RealVal(3.3), RealVal(4.4), RealVal(5.5), RealVal(6.6) ],
       [ RealVal(2.2), RealVal(3.3), RealVal(4.4), RealVal(5.5), RealVal(6.6) ],
       [ RealVal(2.2), RealVal(3.3), RealVal(4.4), RealVal(5.5), RealVal(6.6) ],
       [ RealVal(2.2), RealVal(3.3), RealVal(4.4), RealVal(5.5), RealVal(6.6) ],
       [ RealVal(2.2), RealVal(3.3), RealVal(4.4), RealVal(5.5), RealVal(6.6) ]
     ]
W2 = [ [ RealVal(6.6), RealVal(5.5), RealVal(4.4), RealVal(3.3), RealVal(2.2) ],
       [ RealVal(6.6), RealVal(5.5), RealVal(4.4), RealVal(3.3), RealVal(2.2) ],
       [ RealVal(6.6), RealVal(5.5), RealVal(4.4), RealVal(3.3), RealVal(2.2) ],
       [ RealVal(6.6), RealVal(5.5), RealVal(4.4), RealVal(3.3), RealVal(2.2) ],
       [ RealVal(6.6), RealVal(5.5), RealVal(4.4), RealVal(3.3), RealVal(2.2) ]
     ]

set_option(rational_to_decimal=True)

def vvmul(V1, V2, n):
  res = 0
  for i in range(0, n):
    res += V1[i] * V2[i]
  return res

def vmmul(V, M, O, n):
  res = [None] * 5
  for i in range(0, n):
    tmp = vvmul(M[i], V, n)
    res[i] = If(tmp > 0, O[i] == tmp, O[i] == 0)
  return res

l1_x_cond = vmmul(X, W1, L1X, 5)
out_x_cond = vmmul(L1X, W2, OX, 5)

l1_y_cond = vmmul(Y, W1, L1Y, 5)
out_y_cond = vmmul(L1Y, W2, OY, 5)

input_cond = [ And(0.1 <= Y[i] - X[i], Y[i] - X[i] < 0.5, Y[i] > 0, X[i] > 0) for i in range(5) ]

output_cond = [ (OY[i] - OX[i] > 2) for i in range(5) ]

s = Solver()
s.add(l1_x_cond +
      out_x_cond +
      l1_y_cond +
      out_y_cond +
      input_cond +
      output_cond)

if (s.check() == sat):
  m = s.model()
  print "X", [m.evaluate(X[i]) for i in range(5)]
  print "Y", [m.evaluate(Y[i]) for i in range(5)]
  print "L1-X", [m.evaluate(L1X[i]) for i in range(5)]
  print "L1-Y", [m.evaluate(L1Y[i]) for i in range(5)]
  print "Out-X", [m.evaluate(OX[i]) for i in range(5)]
  print "Out-Y", [m.evaluate(OY[i]) for i in range(5)]
else:
  print s.check()
