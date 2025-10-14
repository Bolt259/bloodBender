#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "write_csv.h"

void writeHeaders(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Time,Glycemia,Insulin Infusion\n";  // Column headers
        file.close();
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}

void appendData(const std::string& filename, double time, double glucose, double insulin_infusion) {
    std::ofstream file(filename, std::ios::app);  // Open in append mode
    if (file.is_open()) {
        file << time << "," << glucose << "," << insulin_infusion << "\n";
        file.close();
    } else {
        std::cerr << "Error opening file for appending!" << std::endl;
    }
}
