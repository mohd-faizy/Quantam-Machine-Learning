# TensorFlow-Quantum

<img src='https://github.com/mohd-faizy/TensorFlow-Quantum/blob/main/TF-Q.png/01_TF-Q.png'>

TensorFlow Quantum (TFQ) is a Python framework for quantum machine learning. As an application framework, TFQ allows quantum algorithm researchers and ML application researchers to leverage Google’s quantum computing frameworks, all from within TensorFlow.


TensorFlow Quantum (TFQ) is a quantum machine learning library for rapid prototyping of hybrid quantum-classical ML models. Research in quantum algorithms and applications can leverage Google’s quantum computing frameworks, all from within TensorFlow

```python

# A hybrid quantum-classical model.
model = tf.keras.Sequential([
    # Quantum circuit data comes in inside of tensors.
    tf.keras.Input(shape=(), dtype=tf.dtypes.string),

    # Parametrized Quantum Circuit (PQC) provides output
    # data from the input circuits run on a quantum computer.
    tfq.layers.PQC(my_circuit, [cirq.Z(q1), cirq.X(q0)]),

    # Output data from quantum computer passed through model.
    tf.keras.layers.Dense(50)
])
```

## __What is a Quantum ML Model?__

> A __Quantum model__ has the ability to represent and generalize data with a quantum mechanical origin. However, to understand quantum models, two concepts must be introduced - quantum data and hybrid quantum-classical models.

Quantum data exhibits superposition and entanglement, leading to joint probability distributions that could require an exponential amount of classical computational resources to represent or store. However, applying quantum machine learning to noisy entangled quantum data can maximize extraction of useful classical information. Inspired by these techniques, the __TFQ__ library provides primitives for the development of models that disentangle and generalize correlations in quantum data, opening up opportunities to improve existing quantum algorithms or discover new quantum algorithms.

The second concept to introduce is hybrid quantum-classical models. __TFQ__ contains the basic structures, such as qubits, gates, circuits, and measurement operators that are required for specifying quantum computations. User-specified quantum computations can then be executed in simulation or on real hardware. Cirq also contains substantial machinery that helps users design efficient algorithms for NISQ machines, such as compilers and schedulers, and enables the implementation of hybrid quantum-classical algorithms to run on quantum circuit simulators, and eventually on quantum processors.

## __TFQ white paper__
[Click here](https://arxiv.org/abs/2003.02989)

<img src='https://github.com/mohd-faizy/TensorFlow-Quantum/blob/main/TF-Q.png/02_TF-Q-P.png'>

## __How TFQ works__

> TFQ allows researchers to construct quantum datasets, quantum models, and classical control parameters as tensors in a single computational graph. Training can be done using standard Keras functions. Just like classical ML, a key challenge of quantum ML is to classify __Noisy Data__.

## __build and train TFQ model__

:heavy_check_mark::one: __Prepare a quantum dataset__ - Quantum data is loaded as tensors . Each quantum data tensor is specified as a quantum circuit written in Cirq that generates quantum data on the fly.

:heavy_check_mark::two: __Evaluate a quantum neural network model__ - The researcher can prototype a quantum neural network using Cirq that they will later embed inside of a TensorFlow compute graph. Parameterized quantum models can be selected from several broad categories based on knowledge of the quantum data's structure. The goal of the model is to perform quantum processing in order to extract information hidden in a typically entangled state.

:heavy_check_mark::three: __Sample or Average__ - Measurement of quantum states extracts classical information in the form of samples from a classical random variable. The distribution of values from this random variable generally depends on the quantum state itself and on the measured observable.

:heavy_check_mark::four: __Evaluate a classical neural networks model__ - As the extracted information may still be encoded in classical correlations between measured expectations, classical deep neural networks can be applied to distill such correlations.

:heavy_check_mark::five: __Evaluate Cost Function__ - This could be based on how accurately the model performs the classification task if the quantum data was labeled, or other criteria if the task is __unsupervised__.

:heavy_check_mark::six: __Evaluate Gradients & Update Parameters__ - After evaluating the cost function, the free parameters in the pipeline should be updated in a direction expected to decrease the cost. This is most commonly performed via __gradient descent__.


---
## Install TensorFlow Quantum

There are a few ways to set up your environment to use TensorFlow Quantum (TFQ):

* The easiest way to learn and use TFQ requires no installation—run the
  [TensorFlow Quantum tutorials](./tutorials/hello_many_worlds.ipynb) directly
  in your browser using
  [Google Colab](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb).
* To use TensorFlow Quantum on a local machine, install the TFQ package using
  Python's pip package manager.
* Or build TensorFlow Quantum from source.

TensorFlow Quantum is supported on Python 3.6, 3.7, and 3.8 and depends directly on [Cirq](https://github.com/quantumlib/Cirq).

## Pip package

### Requirements

* pip 19.0 or later (requires `manylinux2010` support)
* [TensorFlow == 2.3.1](https://www.tensorflow.org/install/pip)

See the [TensorFlow install guide](https://www.tensorflow.org/install/pip) to
set up your Python development environment and an (optional) virtual environment.

Upgrade `pip` and install TensorFlow

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.3.1</code>
</pre>
<!-- common_typos_enable -->

### Install the package

Install the latest stable release of TensorFlow Quantum:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>
<!-- common_typos_enable -->

Success: TensorFlow Quantum is now installed.

Install the latest nightly version of TensorFlow Quantum:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>
<!-- common_typos_enable -->

## Build from source

The following steps are tested for Ubuntu-like systems.

### 1. Set up a Python 3 development environment

First we need the Python 3.8 development tools.
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.8</code>
  <code class="devsite-terminal">sudo apt install python3.8 python3.8-dev python3.8-venv python3-pip</code>
  <code class="devsite-terminal">python3.8 -m pip install --upgrade pip</code>
</pre>
<!-- common_typos_enable -->

### 2. Create a virtual environment

Go to your workspace directory and make a virtual environment for TFQ development.
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.8 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>
<!-- common_typos_enable -->

### 3. Install Bazel

As noted in the TensorFlow
[build from source](https://www.tensorflow.org/install/source#install_bazel)
guide, the <a href="https://bazel.build/" class="external">Bazel</a>
build system will be required.

To ensure compatibility with TensorFlow 2.3.1, we use `bazel` version 3.1.0. To remove any existing version of Bazel:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>
<!-- common_typos_enable -->

Download and install `bazel` version 3.1.0:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel_3.1.0-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_3.1.0-linux-x86_64.deb</code>
</pre>
<!-- common_typos_enable -->

To prevent automatic updating of `bazel` to an incompatible version, run the following:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>
<!-- common_typos_enable -->

Finally, confirm installation of the correct `bazel` version:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel --version</code>
</pre>
<!-- common_typos_enable -->


### 4. Build TensorFlow from source

Here we adapt instructions from the TensorFlow [build from source](https://www.tensorflow.org/install/source)
guide, see the link for further details. TensorFlow Quantum is compatible with TensorFlow version&nbsp;2.3.

Download the
<a href="https://github.com/tensorflow/tensorflow" class="external">TensorFlow source code</a>:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.3.1</code>
</pre>

Be sure the virtual environment you created in step 2 is activated. Then, install the TensorFlow dependencies:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'</code>
  <code class="devsite-terminal">pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">pip install -U keras_preprocessing --no-deps</code>
  <code class="devsite-terminal">pip install numpy==1.18.0</code>
</pre>
<!-- common_typos_enable -->

Configure the TensorFlow build. When asked for the Python interpreter and library locations, be sure to specify locations inside your virtual environment folder.  The remaining options can be left at default values.

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>
<!-- common_typos_enable -->

Build the TensorFlow package:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>
<!-- common_typos_enable -->

Note: It may take over an hour to build the package.

After the build is complete, install the package and leave the TensorFlow directory:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/<var>name_of_generated_wheel</var>.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>
<!-- common_typos_enable -->

### 5. Download TensorFlow Quantum

We use the standard [fork and pull request workflow](https://guides.github.com/activities/forking/) for contributions.  After forking from the [TensorFlow Quantum](https://github.com/tensorflow/quantum) GitHub page, download the source of your fork and install the requirements:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/<var>username</var>/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>
<!-- common_typos_enable -->


### 6. Build the TensorFlow Quantum pip package

Build the TensorFlow Quantum pip package and install:

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/<var>name_of_generated_wheel</var>.whl</code>
</pre>
<!-- common_typos_enable -->

To confirm that TensorFlow Quantum has successfully been installed, you can run the tests:
<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>
<!-- common_typos_enable -->


Success: TensorFlow Quantum is now installed.

> [Source](https://github.com/tensorflow/quantum/blob/master/docs/install.md)
---
 
### Connect with me:


[<img align="left" alt="codeSTACKr | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter]
[<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />][StackExchange AI]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/faizy-mohd-836573122/
[StackExchange AI]: https://ai.stackexchange.com/users/36737/cypher


---


![Faizy's github stats](https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true)


[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=mohd-faizy&layout=compact)](https://github.com/mohd-faizy/github-readme-stats)


