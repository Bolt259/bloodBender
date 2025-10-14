#ifndef RBF_H
#define RBF_H

#define HIDDEN_NEURONS 11
void forward_pass(double control_error, double *activations);
double compute_rbf_output(double *weights, double *activations);
double gaussian_activation(double center, double width, double control_error);
double control_law(double control_error, double target_glycemia, double disturbance, double control_gain, double f_hat, double b_hat);

#endif