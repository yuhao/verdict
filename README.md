# Verdict: Safe and Robust Machine Learning

##Mission Statement
Machine learning has brought lots of excitement and potential for AI technologies to be beneficial for humanity. However, a series of challenges have to be addressed before AI/ML systems can be practically deployed and integrated into our eveyday life. Cutting-edge research challenges include performance, accuracy, scalability, energy-efficiency, privacy, security, etc. While we strongly support research in these areas, this project specifically focuses on the *safty* and *robustness* aspect of machine learning. That is, how do we make sure a machine learning-based system is robust against uncertainty and failure, and will not perform unintended and harmful behaviors? Think about the autopilot system in a [self driving car](http://www.nytimes.com/2016/07/13/business/tesla-autopilot-fatal-crash-investigation.html?_r=0), or the controller of a satellite, or more controversially [predicting how likely someone is going to commit a crime](http://www.bloomberg.com/features/2016-richard-berk-future-crime/).

Our mission is to build systems that enable safe and robust machine learning, thereby pushing machine learning systems one step closer toward reality. By systems, we mean both software systems (e.g., programming language, runtime/middleware, formal verifier) and hardware systems (e.g., customized accelerators, FPGAs).

##State of the Project

As of now, this project has a formal verifier for verifying the *robustness* of a machine learning model. This is a living document of the project. It might be messy and is constantly being updated. More goals will be defined and more systems will be developed as we get a deeper understanding of the problem space.

###What is a Robust Model

Informally, a robust machine learning model should produce same or very close predictions for inputs that are very similar to each other. For example, an image classification model should produce the same label for an image even if the image is slightly perturbated. Formally, we want to make sure:

```
\forany x, |x - x'| < \delta --> |f(x) - f(x')| < \theta
```
where `\delta` and `\theta` are parameters to control the perturbation and robustness, respectively. `-->` denotes the logic implication operator.

Note that this is just one (perhaps most intuitive) definition of robustness. There could be others. For example, one could argue that a robust model should also guarantee `\forany x, |f(x) - f(x')| < \theta --> |x - x'| < \delta`. That is, if two predictions are close enough, their corresponding inputs should not be too different. However, one could further argue that if a model can correctly label two seemingly completely different inputs as the same thing, then that model is robust because it is not easily fooled! See [this](http://arxiv.org/pdf/1412.1897v4.pdf) paper for a concrete example. The exact definition of robustness can vary, and the verifier should be general enough to handle different cases.

###Dependencies
1. [Z3 theorem prover](https://github.com/Z3Prover/z3). Follow its installation instructions. You will need it to run the formal verifier. Remember to build its python binding.
2. [Tensorflow](https://github.com/tensorflow/tensorflow). For training a machine learning model (running applications under `mnist/` for example). Technically this is not a required dependency if you know how to train a machine learning model by yourself or in other languages. The formal verifier only cares about the details of a pre-traind model (e.g., the topology and weights in the case of a neural network) and does not care how it is trained.
3. Python 2.7. We have not tested it on Python 3.x.

###Directory Structure
* `mnist/` contains various tensorflow-based applications for constructing machine learning models for MNIST. The `para/` subdirectory contains model parameters for various models. They will be read by a verifier.
* `simple.py` is a verifier for an artifial 1-layer 5*5 neural net. It's mainly for demonstration purposes.
* `dnn_mnist.py` is a verifier for a 3-layer DNN (784 * 128 * 32 * 10) for MNIST trained by `mnist/fully_connected_feed.py`
* `softmax_mnist.py` is a verifier for a linear softmax model for MNIST trained by `mnist/mnist_softmax.py`

##Readings
See the reading list [page](https://github.com/yuhao/verdict/blob/master/readings.md). There are a wide range of papers discussing the safty and robustness of machine learning in specific applications domains as well as techniques to address them. The list is undoutedly incomplete, and is getting constantly updated. Let me know if you know of a relavant paper!

