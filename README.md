# First-Artificial-Intelligence

## Case Study

For this first experiment we will imagine a small case study. 
I have several objects and I decide to classify them according to their color, blue or red, and of course I took the time to note their length and width.

|        | red | blue | red | blue | red | blue | red | blue | unknown | 
|--------|-----|------|-----|------|-----|------|-----|------|---------|
| length | 3   | 2    | 4   | 3    | 3.5 | 2    | 5.5 | 1    | 4.5     |
| width  | 1.5 | 1    | 1.5 | 1    | 0.5 | 0.2  | 1   | 1    | 1       |

However, it seems that I forgot to note the color of the last object and I only have its length and width.
So I'm going to create an artificial intelligence that will allow me to find the color of my object.

## Instantiation of matrices

We are going to use the numpy library to manage the matrices.  
First we will insert the input values of the synapses.  
```python
# Input data [length, width]
input_data = numpy.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1]), dtype=float)
```
Then, the second array will correspond to the output data, if the result is 1 it's red and if it's 0 then it's blue.  
```python
output_data = numpy.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype=float)
```
Now we will bring everything to the same scale to be able to perform our operations.  
```python
# Allows you to obtain an array of values between 0 and 1
input_data = input_data/numpy.amax(input_data, axis=0)
```
Then we will get our first 8 values to train our AI on them. The last value will not interest us immediately because we will have to guess its value.
```python
data = numpy.split(input_data, [8])[0] # We get the first 8 values of input_data
unknown = numpy.split(input_data, [8])[1] # We get the last value of input_data
```

## Neural network class

We will create our neural network class and we will define our input synapses, output synapses and hidden synapses. 
```python
class Neural_Network(object) :
    def __init__(self) :
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
```
We will also need other parameters, so we will have weight matrices with random weights.
```python
        self.synapse1 = numpy.random.randn(self.inputSize, self.hiddenSize) # (2x3) First weight matrix 
        self.synapse2 = numpy.random.randn(self.hiddenSize, self.outputSize) # (2x1) Second weight matrix
```

## Forward propagation

We will then define our forward propagation function which allows us to multiply our input values by the weight and apply the sigmoid function to the result to obtain our final value.
```python
def forward(self, data) : 
    self.x = numpy.dot(data, self.synapse1) # Matrix multiplication between input values and weights
    self.y = self.sigmoid(self.x) # Application of the activation function, and obtaining the hidden values

    self.z = numpy.dot(self.y, self.synapse2)  # Matrix multiplication between hidden values and weights
    output = self.sigmoid(self.z) # Application of the activation function, and obtaining our final output value
    return output
```
As explained above, we will use the sigmoid function as the activation function for our neural network.
```python
def sigmoid(self, x) :
    return 1/(1 + numpy.exp(-x))
```

## Objective function (Loss function)

However, once the function is executed the results are not good because the weights of our synapses are random.  
So we have to do some reverse propagation to modify our weights and to balance them. We will program an objective function to calculate the error of our network.
```python
def backward(self, data, output_data, output) :
    self.output_error = output_data - output # Calculation of the error
    self.output_delta = self.output_error * self.sigmoidPrime(output) # Application of the derivative of the sigmoid to this error

    self.y_error = self.output_delta.dot(self.synapse2.T) # Calculation of the error of our hidden neurons
    self.y_delta = self.y_error * self.sigmoidPrime(self.y) # Application of the derivative of the sigmoid to this error

    # We adjust our weights  
    self.synapse1 += data.T.dot(self.y_delta)
    self.synapse2 += self.y.T.dot(self.output_delta)
```
And we will add the sigmoidPrime function which is the derivative of the sigmoid function.  
```python
def sigmoidPrime(self, x) :
    return x * (1 - x)
```

## Training function

Finally, we need to create a function to train our neural network and get the most accurate value possible.
```python
def training(self, data, output_data) :
    output = self.forward(data)
    self.backward(data, output_data, output)
```

## Final step

Finally, we will add a prediction function to display the result obtained by our neural network.
```python
# Prediction function
def predict(self):
    
    print("Predicted data after training: \n")

    print("Input : \n" + str(output_data))
    print("Output : \n" + str(numpy.round(NN.forward(data), 2)))

    print("Unknown input : \n" + str(unknown) + "\n")
    print("Unknown output : \n" + str(self.forward(unknown)) + "\n")

    if(self.forward(unknown) < 0.5):
        print("The object is BLUE ! \n")
    else:
        print("The object is RED ! \n")
```
