from z3 import *
import numpy as np
import time

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

def argmax(X):
  maxId = 0
  maxVal = X[0]
  num = len(X)
  for i in range(1, num):
    if simplify(X[i] >= maxVal):
      maxVal = X[i]
      maxId = i
  return maxId

def scaleInput(X, scale, n):
  # x is an Int array, scale is Real, res will be a Real array
  res = [None] * n
  for i in range(n):
    res[i] = X[i] / scale
  return res

# Robustness based on the reference X
def ref_robust(X, Y, n):
  maxId = argmax(X)
  return [ Y[maxId] >= Y[i] for i in range(n) if i != maxId ]

# Full robustness without a reference
def full_robust(X, Y, n, mode, bound=1):
  if mode == "imprecise":
    # This is an imprecise constraint. It asserts that a model is robust if
    # *all* labels' errors are bound by |output_bound|. Depending on the value
    # of |output_bound|, this constraint could be more relaxed or stricter than
    # the precise constraint. Relaxed or strict, it is easier to solve.
    return And( [ And(Y[i] - X[i] < bound, X[i] - Y[i] < bound) for i in range(n) ] )
  else:
    # This is the precise constraint for robustness, but more complex to solve.
    # It is equivalent to assert argmax(X) == argmax(Y). Note that if there are
    # multiple occurrences of the max value, the standard argmax will return
    # only the first occurrence. In that case, this |robust| implementation
    # for checking robustness is incorrect. For example, if X = [4, 4, 4] and Y
    # = [3, 5, 3], then argmax(X) is 0 and argmax(Y) is 1.  In this case,
    # argmax(X) != argmax(Y) while robust(X, Y, 3) is still true because
    # And(X[1] >= X[0], X[1] >= X[2], Y[1] >= Y[0], Y[1] >= Y[2]) == True.
    # See argmax.py for more details.
    return Or( [ And( [ And( X[j] >= X[i], Y[j] >= Y[i] ) for i in range(n) if i != j ] ) for j in range(n) ] )

# To Make sure all elements in X are different
def unique(X, n):
  return And([X[i] != X[j] for i in range(n) for j in range(i) if j != i])

def sigmoid(x):
  natureExp = RealVal(math.e)
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
def vmmul(V, M, B, m, n, act_func):
  res = [None] * n
  for i in range(0, n):
    tmp = vvmul(V, M[i], B[i], m)
    if act_func == "relu":
      res[i] = If(tmp > 0, tmp, 0)
    elif act_func == "reluC":
      reluC = RealVal(0.01)
      res[i] = If(tmp > 0, tmp, tmp * reluC)
    elif act_func == "sigmoid":
      res[i] = sigmoid(tmp)
    elif act_func == "approx_sigmoid":
      res[i] = approx_sigmoid(tmp)
    else:
      res[i] = tmp
  return res

def gaussian_kernel(size, sigma=1):
  x, y = np.mgrid[-size:size+1, -size:size+1]
  g = np.exp(-(x**2 / float(size) + y**2 / float(size))/(2 * sigma**2))
  return g / g.sum()

def convolve(X, K, isize):
  ksize = K.shape[0]
  kprime = ksize / 2

  conv_X = [0] * (isize * isize)

  for i in range(isize):
    for j in range(isize):
      for m in range(ksize):
        for n in range(ksize):
          iprime = i - kprime + m
          jprime = j - kprime + n
          if (iprime < 0 or iprime >= isize or jprime < 0 or jprime >= isize):
            value = 0
          else:
            value = X[iprime * isize + jprime]
          conv_X[i * isize + j] += value * K[m][n]

  return conv_X

def solveIt(s):
  startTime = time.time()
  result = s.check()
  duration = time.time() - startTime
  print "[Solver Runtime] %.2f %s" % (duration, result)

  return result

def genCounterExp(filename, m, X, Y):
  n = len(X)
  with file(filename, 'w') as outfile:
    inx = [convertToPythonNum(m.evaluate(X[i])) for i in range(n)]
    np.savetxt(outfile, np.atleast_2d(inx), delimiter=",")
    iny = [convertToPythonNum(m.evaluate(Y[i])) for i in range(n)]
    np.savetxt(outfile, np.atleast_2d(iny), delimiter=",")

