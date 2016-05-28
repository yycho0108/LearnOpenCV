Convolution Neural Network

- [x] Implemented Convolutional Layer
- [x] Implemented Pooling Layer
- [x] Implemented Activation Layer 
- [x] Implement Neural Network
- [ ] Implement Optimization Schemes other than naive gradient descent
- [ ] Vectorize Layers(Convolution Depths) from std::vector
- [x] Introduce Momentum from previous dw
- [ ] Automatically adjust alpha (learning rate) based on dw

Debugging
- [x] Tested Until al2 Layer --> Now Working
- [x] Tested Conv Layer --> Reasonably Working

To Do:
- [x] Modularize Code: too much in one file
- [x] Test With Learning, not just running
- [x] Logistic Regression / SoftMax Layer
- [x] Solve the "Dead Relu" problem --> solved by initializing to mostly positive weights
- [x] Handle SIGINT to quit midway gracefully
- [x] Add Saving/Loading of Networks
- [x] Turn Hard-coded parameters into tunable parameters
- [x] Add Dropout Units

Dead Relu
![Dead Relu](images/DeadRelu.png)

Alive Relu
![Alive Relu](images/AliveRelu.png)

---
Training Sample With MNIST Data:

Identified Kernels:
![Kernels](images/Kernels.png)

Classification Results:

![Results2](images/Results_2.png)
With better parameters, the ConvNet achieved >95% Classification Success.

Older Result:
![Results](images/Results.png)
