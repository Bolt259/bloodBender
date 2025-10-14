# bareMetalBender 🔧⚙️

**Low-Level C++ Glucose Dynamics Solver**

The embedded systems implementation of the bloodBender prediction engine - a high-performance C++ IVP (Initial Value Problem) solver for real-time glucose dynamics modeling.

## Overview

bareMetalBender is the computational core that powers bloodBender's predictions. Written in C++ for maximum performance, it implements:

- **IVP Solver**: Runge-Kutta methods for differential equation solving
- **RBF Interpolation**: Radial Basis Function approximation for smooth glucose curves
- **Virtual Patient Modeling**: Simulation of glucose-insulin dynamics
- **CSV Output**: Real-time data export for analysis

## Purpose

While bloodBath (Python) handles data synchronization, cleaning, and LSTM model training, bareMetalBender provides:

- ⚡ **Performance**: C++ for real-time computation
- 🎯 **Precision**: Numerical methods for accurate glucose predictions
- 🔌 **Embeddability**: Designed for integration into embedded systems
- 🧮 **Mathematical Core**: Differential equation solving for glucose dynamics

## Components

```
bareMetalBender/
├── ivp.cpp / ivp.h         # IVP solver implementation
├── rbf.cpp / rbf.h         # Radial Basis Function interpolation
├── system.cpp              # Glucose-insulin system dynamics
├── write_csv.cpp / .h      # CSV output utilities
├── Makefile                # Build system
├── plot_data.py            # Python plotting utilities
└── VirtualPatient/         # Test data and virtual patient profiles
    ├── glucose_data.csv    # Sample glucose readings
    └── out.txt             # Solver output
```

## Building

```bash
cd bareMetalBender
make
```

This compiles the IVP solver and generates the `ivp` executable.

## Usage

```bash
# Run the IVP solver
./ivp

# Output will be written to VirtualPatient/out.txt
# Can be visualized with:
python plot_data.py
```

## Integration with bloodBender

bareMetalBender provides the mathematical validation and embedded deployment path for bloodBender's LSTM predictions. While bloodBath handles data processing in Python, bareMetalBender implements real-time glucose dynamics on bare metal hardware.

---

**bareMetalBender** - The iron core of bloodBender's prediction engine 🔧⚙️

_Close to the metal, close to real-time, close to life-saving predictions._
