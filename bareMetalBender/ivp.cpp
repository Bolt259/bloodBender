#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include "ivp.h"

/* Constant system parameters defined by paper under Materials/Methods */
const double tau1 = 49.00;      // pharmacokinetic time constants (min)
const double tau2 = 47.00;      // pharmacokinetic time constants (min)
const double CI = 2010.00;      // insulin clearance (mL/min)
const double p2 = 1.06e-2;      // time constant for insulin action (min^-1)
const double SI = 8.11e-4;      // insulin sensitivity (mL / μU / min)
const double GEZI = 2.20e-3;    // Glucose effectiveness at zero insulin (min^-1)
const double EGP = 1.33;        // endogenous glucose production (mg/dL/min)
const double taum = 40.50;      // the peak time of meal glucose appearance (min)
const double VG = 253.00;       // glucose          ribution volume (dL)
const double time_step = 1.00 / (1000); // time step of 10^-3 min for RK4 method


// helper function to return RA, the rate of glucose appearance (mg/dL/min)
// simulates increase in blood glucose due to food intake
double meal_intake(double CH, double time){
    // equation (2) in paper
    return (CH * time / (VG * taum * taum)) * exp(-1 * time / taum);
}


// calculate time derivatives of 4D vector, given in equation (1)
void compute_derivatives(struct PatientState *patient, double insulin_infusion, double RA, double dYdt[4]){
    // time derivative of subcutaneous insulin concentration
    dYdt[0] = ((-1 * patient->ISC)) / tau1 + ((insulin_infusion) / (tau1 * CI));

    // time derivative of plasma insulin concentration
    dYdt[1] = ((patient->ISC) / tau2) - ((patient->I) / tau2);

    // time derivative of insulin effect on blood glucose
    dYdt[2] = (-1 * p2 * (patient->IEFF)) + (p2 * SI * (patient->I));

    // time derivative of blood glucose concentration
    dYdt[3] = (-1) * (GEZI + (patient->IEFF)) * (patient->G) + EGP + RA;
}


// fills the fill_state structure with the calculations of the 4 differential equations 
void update_temp_state (struct PatientState *base_state, struct PatientState *fill_state, double time_step, double k[4]){
    fill_state -> ISC = base_state -> ISC + ((time_step) * k[0]);
    fill_state -> I = base_state -> I + ((time_step) * k[1]);
    fill_state -> IEFF = base_state -> IEFF + ((time_step) * k[2]);
    fill_state -> G = base_state -> G + ((time_step) * k[3]);
}


// update all patient parameters for the next step in time
void update_patient_state_rk4(struct PatientState *patient, double insulin_infusion, double CH, double meal_timer){
    // simulate meal
    double RA = meal_intake(CH, meal_timer);
    
    // need 4 arrays of 4 components
    // each index in the array corresponds with one of the differential equations
    double k1[4], k2[4], k3[4], k4[4];

    // temporary patient state structure to hold intermediate values
    struct PatientState temp;

    // compute normal derivative to get k1
    compute_derivatives(patient, insulin_infusion, RA, k1);

    // k2 
    update_temp_state(patient, &temp, time_step / 2, k1);
    compute_derivatives(&temp, insulin_infusion, RA, k2);

    // k3
    update_temp_state(patient, &temp, time_step / 2, k2);
    compute_derivatives(&temp, insulin_infusion, RA, k3);

    // k4
    update_temp_state(patient, &temp, time_step, k3);
    compute_derivatives(&temp, insulin_infusion, RA, k4);

    // calculate the values for the 4D vector field at t = time + time_step
    patient -> ISC = (patient -> ISC) + ((time_step / 6) * (k1[0] + (2 * k2[0]) + (2 * k3[0]) + k4[0])); 
    patient -> I = (patient -> I) + ((time_step / 6) * (k1[1] + (2 * k2[1]) + (2 * k3[1]) + k4[1])); 
    patient -> IEFF = (patient -> IEFF) + ((time_step / 6) * (k1[2] + (2 * k2[2]) + (2 * k3[2]) + k4[2])); 
    patient -> G = (patient -> G) + ((time_step / 6) * (k1[3] + (2 * k2[3]) + (2 * k3[3]) + k4[3])); 
}

double adjust_insulin(double G, double target_G, double current_insulin) {
    const double Kp = 0.001;  // Proportional gain (μU per mg/dL error)
    double error = G - target_G;  // Difference from target glucose

    // Proportional control: Increase/decrease insulin based on glucose level
    double insulin_infusion = current_insulin + (Kp * error);

    // Ensure insulin stays within safe limits (μU/min)
    if (insulin_infusion < 0) insulin_infusion = 0;        // No negative insulin
    if (insulin_infusion > 30000) insulin_infusion = 30000;  // Upper safety limit

    return insulin_infusion; 
}





