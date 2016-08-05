BACKPROPAGATION ALGORITHM TO PERFORM XOR OPERATION


Architecture :

The neural network is a 3 layer network; input layer, 1 hidden layer and output layer. Input layer takes in the binary values. There are 3 nodes present in the input layer: 2 binary values and the bias term. Similarly the hidden layer contains 3 nodes : 2 values and the bias term. The output layer has 1 neuron.  

Variables used :

layer1                       = variable for input values
target_output                = variable that contains the proper outputs
weights_1                    = matrix of weights for the first layer
weights_2                    = matrix of weights for the second layer
a2, a3                       = activations of the hidden layer and the final layer
a2_error, a3_error           = Error in the 2nd and 3rd layer
a2_delta, a3_delta           = Calculating the gradient between activation of a particular layer and weights 	                                                       associated with it.


Working :

1. layer1 = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    	- Initialising the inputs for the 1st layer.

2. target_output = np.array([[0, 1, 1, 0]]).T
	- Fixing the outputs that we want out net to achieve.

3.  weights_1 = np.random.random((3, 2))
  weights_2 = np.random.random((3, 1))
      - Initialising the weight matrices for the hidden and output layer.

4. def sigmoid(g):
    return 1/(1 + np.exp(-2*g))
      - Definition of the sigmoidal function that calculates the activation of the neurons.

5. def sigmoid_gradient(g):
    return g*(1 - g)
      - Definition of the function that calculates the derivative of the sigmoidal function.

6. a2 = sigmoid(np.dot(layer1, weights_1))
      - Calculating the activation values by multiplying the inputs and weights associated with the input layer.

7. a2 = a2.T
   a2 = np.vstack((a2, bias)).T
      - Adding an extra row of bias values to the activation matrix.

8. a3 = sigmoid(np.dot(a2, weights_2))
      - Calculating the activation values of the final layer by multiplying the activation values of the hidden layer and weight matrix associated with the hidden layer.

9. a3_error = target_output - a3
   a2_error = np.dot(a3_error, weights_2[0:2, :].T)*sigmoid(np.dot(layer1, weights_1))
      - Calculating the error of the final layer and using that error, we calculate the error of the hidden layer.

10. a3_delta = a3_error*sigmoid_gradient(a3)
    a2_delta = a2_error*sigmoid_gradient(a2[:, 0:2])
      - Calculating the gradient of the error terms. This expression gives us the relation between the error terms and the neurons of a particular layer.

11. weights_2 += np.dot(a2.T, a3_delta)
    weights_1 += np.dot(layer1.T, a2_delta)
      - Updating the values of weights of hidden layer and output layer.

Steps 6 through 11 are repeated for 100000 times. With every iteration, the cost reduces as the weights keep updating.


Output : 

After training : 
[[ 0.00253896]
 [ 0.99785759]
 [ 0.99785759]
 [ 0.00219845]]

Process finished with exit code 0
