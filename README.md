# Verdict: Safe and Robust Machine Learning

##Mission Statement
Machine learning has brought lots of excitement and potential for AI technologies to be beneficial for humanity. However, a series of challenges have to be addressed before AI/ML systems can be practically deployed and integrated in our eveyday life. Cutting-edge research challenges include performance, accuracy, scalability, energy-efficiency, privacy, security, etc. While we strongly support research in these areas, this project specifically focus on the *safty* and *robustness* aspect of machine learning. That is, how do we make sure a machine learning-based system is robust against uncertainty and failure, and will not perform unintended and harmful behaviors? Think about the autopilot system in a [self driving car](http://www.nytimes.com/2016/07/13/business/tesla-autopilot-fatal-crash-investigation.html?_r=0) or the controller of a satellite.

Our mission is to build systems that enable safe and robust machine learning, thereby pushing machine learning systems one step closer toward reality. By systems, we mean both software systems (e.g., programming language, runtime/middleware, formal verifier) and hardware systems (e.g., customized accelerators, FPGAs).

##State of the Project

As of now, this project has a formal verifier for verifying the *robustness* of a machine learning model. This is a living document of the project. It might be messy and is constantly being updated. More goals will be defined and more systems will be developed as we get a deeper understanding of the problem space.

###What is a Robust Model

Informally, a robust machine learning model should produce same or very close predictions for inputs that are very similar to each other. For example, an image classification model should produce the same label for an image even if the image is slightly perturbated. Formally, we want to make sure:

```
\forany x, |x - x'| < \delta --> |f(x) - f(x')| < \theta
```
where `\delta` and `\theta` are parameters to control the perturbation and robustness, respectively. `-->` denotes the logic implication operator.

Note that this is just one intuitive definition of robustness. One could argue that a robust model should also guarantee `\forany x, |f(x) - f(x')| < \theta --> |x - x'| < \delta`. That is, if two predictions are close enough, their corresponding inputs should not be too different. However, one could further argue that if a model can correctly label two seemingly completely different inputs as the same thing, then that model is robust because it is not easily fooled! See [this](http://arxiv.org/pdf/1412.1897v4.pdf) paper for a concrete example. The exact definition of robustness can vary, and the verifier should be general enough to handle different cases.

###Dependencies
1. [Z3 theorem prover](https://github.com/Z3Prover/z3). Follow its installation instructions. You will need it to run the formal verifier. Remember to build its python binding.
2. [Tensorflow](https://github.com/tensorflow/tensorflow). For training a machine learning model (running applications under `mnist/` for example). Technically this is not a required dependency if you know how to train a machine learning model by yourself or in other languages. The formal verifier only cares about the details of a pre-traind model (e.g., the topology and weights in the case of a neural network) and does not care how it is trained.
3. Python 2.7. We have not tested it on Python 3.x.

###Directory Structure
* `mnist/` contains various tensorflow-based applications for constructing machine learning models for MNIST. The `para/` subdirectory contains model parameters for various models. They will be read by a verifier.
* `simple.py` is a verifier for an artifial 1-layer 5*5 neural net. It's mainly for demonstration purposes.
* `dnn_mnist.py` is a verifier for a 3-layer DNN (784 * 128 * 32 * 10) for MNIST trained using `mnist/fully_connected_feed.py`
* `softmax_mnist.py` is a verifier for a linear softmax model for MNIST trained using `mnist/mnist_softmax.py`

###Usage
1. `python dnn_mnist.py` launches the DNN verifier. Use `-h` to get commandline options.
  * `-i`: input perturbation, choose between (0, 1). By default, it is set to `0.0001`, indicating that the acceptable perturbation in the (scaled) pixel intensity is below `0.001`.
  * `-r`: robustness constraint, choose between "imprecise, precise". By default, it is set to "precise", which is the most precise robustness constraints. In a classification model, it forces Z3 to verify `argmax(f(X)) == argmax(f(X'))`, where `X` and `X'` are the original input and perturbed input. The "imprecise" constraint is imprecise in the sense that it could be more relaxed or stricter than the precise one. But "imprecise" takes less time to verify. The runtime between the two could be orders of magnituide different.
  * `-a`: activation function, choose between "none, relu, reluC, sigmoid, approx_sigmoid". By default it's set to "none". Setting it to "relu" is ideally what we want to verify, but the number of constraints grows exponentially because the ReLU function introducs a logic *OR*. The verifier is not complete if one uses "sigmoid" because Z3 does not have a complete solver for non-polynomial real arithmetic. If you really want to use the sigmoid activation function, you could use "approx_sigmoid", which uses the first 4 terms of sigmoid's taylor series to approaximate sigmoid (which is what a software library or hardware implementation will do anyways). "approx_sigmoid" transforms a set of exponential constraints to a set of polynomial constraints, for which Z3 has a complete solver in the real domain.
2. `python softmax_mnist.py` launches the Softmax verifier. The usage is similar to `dnn_mnist.py` except there is no `-a` option because a softmax model does not have an activation function.
  * Just to provide an intuition of the runtime difference between different robustness constraints, on a Xeon E5-2640 CPU running at 2.50GHz, the "imprecise" constraint took about 24 mins, and the "strong" constraint took about 40 mins. MNIST only has 10 labels--the runtime difference grows exponentially as the number of output labels increases (Take a look at their logic formulars to understand why).

##Readings

###Position Papers and Articles
1. [Concrete Problems in AI Safety](https://arxiv.org/pdf/1606.06565v1.pdf)
2. [Machine Learning: The High-Interest Credit Card of Technical Debt](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43146.pdf)
3. [Cyber-physical systems you can bet your life on](https://www.microsoft.com/en-us/research/cyber-physical-systems-can-bet-life/)

###Technical Papers
There are a wide range of papers discussing the safty and robustness of machine learning in specific applications domains as well as techniques to address them. The following is undoutedly an incomplete list, and is getting constantly updated. Let me know if you know of a relavant paper!

1. [Intriguing Properties of Neural Networks](https://cs.nyu.edu/~zaremba/docs/understanding.pdf)
2. [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](http://arxiv.org/pdf/1412.1897v4.pdf)
3. [Improving the Robustness of Deep Neural Networks via Stability Training](http://arxiv.org/pdf/1604.04326v1.pdf)
4. [Towards Deep Neural Network Architectures Robust to Adversarial Examples](http://arxiv.org/pdf/1412.5068v4.pdf)
5. [Measuring Neural Net Robustness with Constraints](http://arxiv.org/pdf/1605.07262v1.pdf)
