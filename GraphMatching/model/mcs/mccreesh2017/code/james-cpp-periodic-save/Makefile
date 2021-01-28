CXX := g++
CXXFLAGS := -g -march=native
all: mcsp

mcsp: mcsp.c graph.c graph.h
	$(CXX) $(CXXFLAGS) -Wall -std=c++11 -o mcsp graph.c mcsp.c -pthread -lboost_filesystem -lboost_system