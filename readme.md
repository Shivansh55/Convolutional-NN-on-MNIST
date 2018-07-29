In this project I have implemented Convolutinal Neural Network on MNIST data set
MNIST is a computer vision data set containing images of handwritten digits.
he input images x will consist of a 2d tensor of floating point numbers. we initialize both W and b as tensors full of zeros. W is a 784x10 matrix (because we have 784 input features and 10 outputs) and b is a 10-dimensional vector (because we have 10 classes).
tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes, and tf.reduce_mean takes the average over these sums.
One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons". 
It will consist of convolution, followed by max pooling. The convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.
 image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
 To reduce overfitting, we will apply dropout before the readout layer.
 We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
 
