'''
.. _Inspired by:
    https://www.youtube.com/watch?v=PwAGxqrXSCs
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0-9
n_classes = 10
'''
One hot means that the labels will be represented on the following manner:
0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
4 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
8 -> [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
'''

# We'll be using 3 hidden layers, each one with 500 perceptrons
'''
Each layer is represented by two matrix: one containing the connection weights
and one containing the biases.

Each layer will multiply the output of the previous layer (usually an np.array
with the shape [1 x n]) by this layer's weights (usually [n x n_of_nodes])) and
add a bias (np.array [1 x n_of_nodes])
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# We will load the data 100 by 100 images. This is very important when whe have
# more data than RAM
batch_size = 100

# Now we're instantiating the objects that will recieve our inputs (the data
# itself).
# Here the data is a float 28x28 np.array, but considering that these
# placeholders will represent the input layer of our neural network we
# will flatten those images so they can be transformed into an 1 x 784 array.
x = tf.placeholder('float', [None, 28 * 28])

# TODO add an explenation
y = tf.placeholder('float')


def neural_network_model(data):
    '''Creates and connects the desired neural network.

    This method is responsible for creating and connecting the layers or our
    neural network.

    Arguments:
        data (tf.placeholder): The input data placeholder.

    Retuns:
        The Tensor object representing the last layer of the neural network.

    '''

    '''
    As I've written at the beginning, the weights are nothing more than
    matrices with specific sizes.

    During the tensorflow session the tf will manipulate these matrices (and
    apparently this manipulation is not done on the Python environment),
    therefore these matrices must be instantiated as an tf.Variable.

    Here we're starting these matrices with a normally distributed random
    values.
    '''
    # For pratical porpuses we will store each layer's variable on a dictionary
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([28 * 28,
                                                               n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,
                                                               n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,
                                                               n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,
                                                             n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    '''
    Now we're creating the layers themselves. As I sad before, each layer is
    the result of the multiplication of the results of the previous layer by
    this layer's weights plus it's biases.
    '''
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),
                hidden_1_layer['biases'])

    '''
    As for the perceptron activation function we will use the ReLU (Rectifier
    Linear Unit), witch, as far as I know, is nothing more than the product of
    a Heaviside and a ramp function.
    '''
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),
                hidden_2_layer['biases'])

    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),
                hidden_3_layer['biases'])

    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']),
                    output_layer['biases'])

    return (output, [hidden_1_layer, hidden_2_layer, hidden_3_layer,
                     output_layer])


def train_neural_network(data_placeholder, label_placeholder):
    '''Instantiates and train a neural network with a given dataset.

    Arguments:
        dataset (tf.placeholder): The input data placeholder.

    '''
    # Creating the neural network and storing its output layer (A [1 x 784]
    # Tensor object)
    prediction, layers = neural_network_model(data_placeholder)

    '''
    Now we're defining the cost function.

    The workflow of the nn training is that it will run all data on a given
    batch through it and will store all the result on an array.

    Once the batch is finished and we alreay have the resulting outputs of our
    neural network we will compare them to the expected results. The higher the
    difference between the predicted and the expected the worst our nn
    perform.

    Here we're defining the metric of this difference, which will be the the
    average of the cross entropy between the actual and the expected result.
    '''
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                labels=label_placeholder))

    '''
    Now we're defining the optimation algorithim.

    We will use the Adam Stochactic gradient-based optmization method.

    As you may see, we're simultaneously instantiatin the AdamOptimizer object
    and defining the `optimizer` variable as the minimization of the cost
    function using this object.
    '''
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # An epoch is a complete cycle of forward propagation and backpropagation
    # of all the data in the training database
    n_epochs = 10

    '''
    Now we're going to start a tensorflow session. Basically, once we've
    defined all the variables, connections and data of our nn we will start the
    'tensorflow engine', that will do all the needed operation on the
    background (apparently outside the python environment).

    Therefore the tensorflow session should be treated as an file, that must be
    opend and closed.
    '''
    with tf.Session() as sess:
        # Starting the tf session and initializing the tf variables
        sess.run(tf.global_variables_initializer())
        result = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # The next_batch returns a list containing both the data and
                # the real output
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                '''
                The Session.run() method  runs one "step" of TensorFlow
                computation, by running the necessary graph fragment to execute
                every Operation and evaluate every Tensor in fetches,
                substituting the values in feed_dict for the corresponding
                input values.

                In this case we're ignoring the output of the `optimizer`
                method amd only retrieving the result of the cost function for
                the specific batch so we can display it.
                '''
                _, c = sess.run([optimizer, cost],
                                feed_dict={data_placeholder: batch_x,
                                           label_placeholder: batch_y})

                epoch_loss += c

            print('Epoch', epoch + 1, 'completed out of', str(n_epochs) +
                  '. Loss:', epoch_loss)

        '''
        Each position of the prediction Tensor contains the estimated
        probability of the data belonging to a given group.

        We get the index of the item with the gratest probability and consider
        it as the estimated group, then we compare it with actual label, which
        generates another array containing all the instances into which we
        predicte correctly.
        '''
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        '''
        Casting the `correct` Tensor from boolean to float (True -> 1.,
        False -> 0.) then calculating the average of this tensor, which will
        yield the porcentage of instance that the nn predicted correctly.
        '''
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy: ',
              str(accuracy.eval({data_placeholder: mnist.test.images,
                                 label_placeholder: mnist.test.labels})))

        '''
        Concatenating the weights of the layers on the result array.

        To get the contents of the tf.Variable tensors we need to use the
        `eval` method, which will return the respective np.array.
        '''
        for tensor in layers:
            result.append({'weights': tensor['weights'].eval(session=sess),
                           'biases': tensor['biases'].eval(session=sess)})

    return result


result_layers = train_neural_network(x, y)
