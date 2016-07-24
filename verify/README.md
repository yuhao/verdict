# Formal Verification of Robustness

##Files
1. `dnn_mnist.py`: verifies whether a given DNN-based model is robust against a given perturbation `\epsilon`.
2. `dnn_mnist_ref.py`: verifies whether a given DNN-based model is robust against a given perturbation `\epsilon` _for a particular input_. This verifier incrementally feeds constraints to the solver to optimize common case performance.
3. `softmax_mnist.py`: verifies whether a given softmax-based model is robust against a given perturbation `\epsilon`.
4. `softmax_mnist_conv.py`: verifies whether a given softmax-based model is robust against Gaussian blur.
5. `verdict_core.py`: provides base functions to all verifiers.
6. `argmax.py`: compares two different ways of expressing `argmax`.

##Usage
1. `python dnn_mnist.py` launches the DNN verifier. Use `-h` to get commandline options.
  * `-i`: input perturbation, choose between (0, 1). By default, it is set to `0.001`, asserting that the acceptable perturbation in the (scaled) pixel intensity is below `0.001`.
  * `-r`: robustness constraint, choose between "imprecise, precise". By default, it is set to "precise". In a model trained for classification, the "precise" constraint forces Z3 to verify `argmax(f(X)) == argmax(f(X'))`, where `X` and `X'` are the original input and perturbed input, respectively. The "imprecise" constraint is imprecise in the sense that it could be more relaxed or stricter than the precise one. The trade-off here is that "imprecise" takes less time to verify. The runtime between the two robustness constraints could be orders of magnituide different.
  * `-m`: verification mode, choose between "general, specific". By default, it is set to "general". The "general" mode verifies the robustness of the model against *any* possible inputs. The "specific" mode verifies the robustness against a specific input (called reference input). Currently, the reference input in the "specific" mode is chosen from MNIST's test data set using the `-c` option (between 0 and 9999). The "specific" mode is must faster than "general" (over 10X) because the former reduces the number of unknown variables.
  * `-a`: activation function, choose between "none, relu, reluC, sigmoid, approx_sigmoid". By default it's set to "none". "relu" is the most widely used activation function, but the number of constraints grows exponentially because the ReLU function introducs a logic *OR*. The verifier is not complete if one uses "sigmoid" because Z3 does not have a complete solver for non-polynomial real arithmetic. If you really want to use the sigmoid activation function, you could try "approx_sigmoid", which uses the first 4 terms of sigmoid's taylor series to approaximate sigmoid (which is what a software library and most hardware implementations will do anyways). "approx_sigmoid" transforms a set of exponential constraints to a set of polynomial constraints, for which Z3 has a complete solver in the real domain.
2. `python softmax_mnist.py` launches the Softmax verifier. The usage is similar to `dnn_mnist.py` except there is no `-a` option because a softmax model does not have an activation function.
  * Just to provide an intuition of the runtime difference between different robustness constraints, on a Xeon E5-2640 CPU running at 2.50GHz, the "imprecise" constraint took about 24 mins, and the "strong" constraint took about 40 mins. MNIST only has 10 labels--the runtime difference grows exponentially as the number of output labels increases (Take a look at their logic formulars to understand why).

