from z3 import *

X = [ Int('x%s' % i) for i in range(3) ]
Y = [ Int('y%s' % i) for i in range(3) ]

# Classical definition of argmax. If multiple maximum exist, return the first one
def argmax(X):
  return If(And(X[0] >= X[1], X[0] >= X[2]), 0, If(And(X[1] >= X[0], X[1] >= X[2]), 1, 2))

def argmax2(X, Y):
  return Or( [ And( [ And( X[j] >= X[i], Y[j] >= Y[i] ) for i in range(3) if i != j ] ) for j in range(3) ] )

def unique(X):
  return And([X[i] != X[j] for i in range(3) for j in range(i) if j != i])

robust = argmax(X) == argmax(Y)
robust2 = argmax2(X, Y)

s = Solver()

# Verify: robust2 --> robust
# Result: sat. Thus robust2 doesn't imply robust
s.push()
s.add(Not(robust), robust2)
res = s.check()
print res

if (res == sat):
  print s.model()

# Verify: robust2 --> robust if X and Y are unique
# Result: unsat. Thus robust2 does imply robust if X and Y are unique
s.push()
s.add(unique(X), unique(Y))
res = s.check()
print res

if (res == sat):
  print s.model()
s.pop()
s.pop()

# Verify: robust --> robust2
# Result: unsat. Thus robust does imply robust2 even if X and Y are not unique
s.add(robust, Not(robust2))
res = s.check()
print res

if (res == sat):
  print s.model()

# Verify: robust --> robust2 if X and Y are unique
# Result: unsat. Thus robust does imply robust2 if X and Y are unique
s.push()
s.add(unique(X), unique(Y))
res = s.check()
print res

if (res == sat):
  print s.model()
s.pop()

###Conclusion###
###robust <==> robust2 only if X and Y are unique, which means we can verify either robust or robust2 interchangeably
###If X and Y are not unique, we only have robust2 --> robust
