#include <iostream>
#include <vector>
#include <cmath> 
#include <cstdlib> 
#include <ctime>   

#include "rbf.h"

// preset centers and widths for each of the hidden neurons
double centers[HIDDEN_NEURONS] = {-25.0, -10.0, -8.0, -5.0, -2.0, 0.0, 5.0, 10.0, 20.0, 40.0, 100.0};
double widths[HIDDEN_NEURONS] = {20.0, 15.0, 10.0, 8.0, 5.0, 5.0, 5.0, 20.0, 40.0, 100.0, 200.0};


// gaussian activation function for each neuron in the network
double gaussian_activation(double center, double width, double control_error)
{
    return exp(-0.5 * (pow(((control_error - center) / width), 2)));
}

// output of the rbf is a weighted sum of all the activation results
double compute_rbf_output(double *weights, double *activations)
{
    double sum = 0;
    for (int i = 0; i < HIDDEN_NEURONS; i++){
        // add the product of weight and activation output to the sum
        sum += weights[i] * activations[i];
    }
    return sum;
}


// calculate the results of the activation function for each neuron
void forward_pass(double control_error, double *activations)
{
    for (int i = 0; i < HIDDEN_NEURONS; i++){
        activations[i] = gaussian_activation(centers[i], widths[i], control_error);
    }
}

// using current error and activation function outputs, update the weights for each neuron
void update_weights(double learning_rate, double control_error, double *activations, double *weights)
{
    // cycle through all neurons and update weights
    for (int i = 0; i < HIDDEN_NEURONS; i++){
        weights[i] = learning_rate * control_error * activations[i];
    }
}

// calculate the insulin infusion rate
double control_law(double control_error, double target_glycemia, double disturbance, double control_gain, double f_hat, double b_hat)
{
    return ((1 / b_hat) * ((-1 * f_hat) + target_glycemia - (control_gain * control_error) - disturbance));
}