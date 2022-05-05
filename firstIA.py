import numpy

# Input data [length, width]
input_data = numpy.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1]), dtype=float)
# Output data 1 = Red / 0 = Blue
output_data = numpy.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype=float)

# Allows you to obtain an array of values between 0 and 1
input_data = input_data/numpy.amax(input_data, axis=0)

data = numpy.split(input_data, [8])[0] # We get the first 8 values of input_data
unknown = numpy.split(input_data, [8])[1] # We get the last value of input_data

# Neural network class
class Neural_Network(object) :
    def __init__(self) :
        # Network settings
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # Random generation of weights 
        self.synapse1 = numpy.random.randn(self.inputSize, self.hiddenSize) # (2x3) First weight matrix 
        self.synapse2 = numpy.random.randn(self.hiddenSize, self.outputSize) # (2x1) Second weight matrix

    # Activation function
    def sigmoid(self, x) :
        return 1/(1 + numpy.exp(-x))

    # Forward propagation function
    def forward(self, data) : 
        self.x = numpy.dot(data, self.synapse1) # Matrix multiplication between input values and weights
        self.y = self.sigmoid(self.x) # Application of the activation function, and obtaining the hidden values

        self.z = numpy.dot(self.y, self.synapse2)  # Matrix multiplication between hidden values and weights
        output = self.sigmoid(self.z) # Application of the activation function, and obtaining our final output value
        return output
    
    # Derivative of the activation function
    def sigmoidPrime(self, x) :
        return x * (1 - x)
    
    # Back propagation function (Backward)
    def backward(self, data, output_data, output) :
        self.output_error = output_data - output # Calculation of the error
        self.output_delta = self.output_error * self.sigmoidPrime(output) # Application of the derivative of the sigmoid to this error

        self.y_error = self.output_delta.dot(self.synapse2.T) # Calculation of the error of our hidden neurons
        self.y_delta = self.y_error * self.sigmoidPrime(self.y) # Application of the derivative of the sigmoid to this error

        # We adjust our weights  
        self.synapse1 += data.T.dot(self.y_delta)
        self.synapse2 += self.y.T.dot(self.output_delta)
    
    def training(self, data, output_data) :
        output = self.forward(data)
        self.backward(data, output_data, output)

    # Prediction function
    def predict(self):
        
        print("Predicted data after training: \n")
        print("Input : \n" + str(unknown) + "\n")
        print("Output : \n" + str(self.forward(unknown)) + "\n")

        if(self.forward(unknown) < 0.5):
            print("The object is BLUE ! \n")
        else:
            print("The object is RED ! \n")

NN = Neural_Network()    

for i in range(1000): 
    NN.training(data, output_data)

NN.predict()
