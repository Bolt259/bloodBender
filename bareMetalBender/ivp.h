#ifndef IVP_H
#define IVP_H

struct PatientState {
    double ISC;                 // subcutaneous insulin concentration (μU/mL)
    double I;                   // plasma insulin concentration (μU/mL)
    double IEFF;                // insulin effect on blood glucose (min^-1)
    double G;                   // blood glucose concentration (mg/dL)
};

double meal_intake(double CH, double time);
void compute_derivatives(struct PatientState *patient, double insulin_infusion, double RA, double dYdt[4]);
void update_temp_state (struct PatientState *base_state, struct PatientState *fill_state, double time_step, double k[4]);
void update_patient_state_rk4(struct PatientState *patient, double insulin_infusion, double CH, double meal_timer);
double adjust_insulin(double G, double target_G, double current_insulin);



#endif