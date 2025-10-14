# Compiler
CXX = g++

# Compiler flags: 
# -std=c++11: use the C++11 standard (or change as needed)
# -Wall: enable all warnings
# -O2: optimization level 2 (optional)
CXXFLAGS = -std=c++11 -Wall -O2

# The target executable name
TARGET = system

# The object file (compiled from ivp.cpp)
OBJS = system.o rbf.o ivp.o write_csv.o

# Default rule: build the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

system.o: system.cpp
	$(CXX) $(CXXFLAGS) -c system.cpp

rbf.o: rbf.cpp
	$(CXX) $(CXXFLAGS) -c rbf.cpp

# Rule to compile ivp.cpp into ivp.o
ivp.o: ivp.cpp
	$(CXX) $(CXXFLAGS) -c ivp.cpp

write_csv.o: write_csv.cpp
	$(CXX) $(CXXFLAGS) -c write_csv.cpp


# Clean rule: remove the executable and object files
clean:
	rm -f $(TARGET) $(OBJS)