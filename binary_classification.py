import numpy as np
import math

class LossFunction:
    def __init__(self):
        self._result = None

    def zero_one(self, feature_vector, weight_vector, y):
        score = np.dot(feature_vector, weight_vector) # how confident we are in prediction
        margin = score * y # how correct weare in prediction
        # the classifier makes an error when margin is less than 0 because the score and y are different signs
        if(np.sign(margin) == -1): self._result = 1
        elif(np.sign(margin) == 1): self._result = 0
        return self._result

# def feature_map(data_point):
#     # returns a feature vector of R superscript d numbers
#     return True

# TODO: synthesise training data
# TODO: write gradient descent algorithm

feature_vector = np.array([1, 2, 3, 4])
weight_vector = np.ones(len(feature_vector))

# The score represents the weighted combination of features and weights
score = np.dot(feature_vector, weight_vector)

def linear_classifier(score): # or the sign(dot(feature_vector, weight_vector))
    return np.sign(score)

def linear_regression(feature_vector, weight_vector): # in some senses this is simpler as it is already R
    return np.dot(feature_vector, weight_vector)

def least_squares(feature_vector, weight_vector, y):
    prediction = linear_regression(feature_vector, weight_vector)
    residual = prediction - y # i.e. the amount of overshoot there is
    loss = (residual * residual) # squared to allow punishment of over or undershoot
    return loss

training_data = [(1, 2), (3, 4)]

def least_squares(w):
    return sum((np.dot(x, w) - y) ** 2 for x, y in training_data)

def dleast_squares(w):
    return sum(2*(np.dot(x, w) - y) * x for x, y in training_data)

# def data_synthesis(num, dim)

def main():
    print("Feature vector: ", feature_vector)
    print("Weight vector: ", weight_vector)
    print("The score: %d" % score)
    print("The classification: %d" % linear_classifier(score))


    loss = LossFunction()
    result = loss.zero_one(feature_vector, weight_vector, -1)
    print("The loss: %d" % result)

    # Gradient descent...

    dim = 1
    w = 0 # weights with 1 dimenson
    step_size = 0.01
    training_data = [(2, 4), (4, 2)]

    for t in range(100):
        value = least_squares(w)
        gradient = dleast_squares(w)
        w = w - step_size * gradient
        print('iteration {}: w = {}, loss = {}'.format(t, w, value))

if __name__ == "__main__":
    main()
