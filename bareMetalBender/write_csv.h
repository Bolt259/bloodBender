#ifndef WRITE_CSV_H 
#define WRITE_CSV_H

#include <string>

// Function to write headers to CSV
void writeHeaders(const std::string& filename);

// Function to append data to CSV
void appendData(const std::string& filename, double time, double glucose, double insulin_infusion);

#endif 
