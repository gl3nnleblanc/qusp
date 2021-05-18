<img src="/dev/svd.svg" width="100%" height="150px">

# PDRP 2021 - QuSP

**Qu**antum **S**imulation **P**layground

[![Build Status](https://www.travis-ci.com/gl3nnleblanc/pdrp2021.svg?branch=master)](https://www.travis-ci.com/gl3nnleblanc/pdrp2021)
## Overview

 This is a sanbox for the physics [directed reading program](https://berkeleyphysicsdrp.wixsite.com/physicsberkeleydrp) during Spring '21. The focus of this code is on quantum simulation with matrix product states. A final report is available under `report/MPS.pdf`.

 Check out the `examples` folder for some neat showcase notebooks!
## Features

* An implementation of matrix product states, supporting
	- constructing one from any initial tensor and specified bond dimension
	- contraction back to a tensor of high (tensor-)rank
	- time evolving block decimation

## Sources

1. [Efficient classical simulation of slightly entangled quantum computations](https://arxiv.org/pdf/quant-ph/0301063.pdf)
1. [TensorNetwork for Machine Learning](https://arxiv.org/pdf/1906.06329.pdf)
1. [Towards Quantum Machine Learning with Tensor Networks](https://arxiv.org/pdf/1803.11537.pdf)
