import numpy as np

STUDENT={'name': 'Peleg shefi_Daniel bazar',
         'ID': '316523638_314708181'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W, b = params
    z = np.dot(x, W) + b
    probs = softmax(z)
    return probs


def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W, b = params
    z = np.dot(x, W) + b
    y_hat = softmax(z)
    loss = -1 * np.log(y_hat[y])
    y_true = np.zeros(y_hat.shape[0])
    y_true[y] = 1
    x_copy = np.array(x)

    x_copy = x_copy.reshape(x_copy.shape[0], 1)
    delta_y = (y_hat - y_true).reshape(1, (y_hat - y_true).shape[0])
    gW = np.dot(x_copy, delta_y)
    gb = y_hat - y_true
    return loss, [gW, gb]


def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    epsilon = np.sqrt(6.0) / np.sqrt(out_dim + in_dim)
    W = np.random.uniform(-epsilon, epsilon, size=(in_dim, out_dim))
    epsilon = np.sqrt(6.0) / np.sqrt(out_dim + 1)
    b = np.zeros((out_dim))

    # W = np.zeros((in_dim, out_dim))*0.01
    # b = np.zeros(out_dim)*0.01
    return [W, b]


if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1, 2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001, 1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001, -1002]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b = create_classifier(3, 4)


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[1]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
