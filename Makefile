CXX = g++
SRC = src
INCLUDE = inc
CXXFLAGS = -std=c++11  -Wall -ggdb $(shell pkg-config --cflags opencv4)
LDLIBS= $(shell pkg-config --libs opencv4)
LIBRARIES = 
EXECUTABLE = main

all: $(EXECUTABLE)
run : clean all
		clear
		./$(EXECUTABLE)

$(EXECUTABLE) : $(SRC)/*cpp
		$(CXX) $(CXXFLAGS) -I$(INCLUDE) $^  $(LDLIBS) -o $@

clean :
		-rm *main


