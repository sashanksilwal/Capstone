CFLAGS = -Wall -std=c++17 -Iinclude
LIBS =  lib/libonnxruntime.dylib

CXX = g++
RM = rm -f

SRCS = src/main.cpp src/helper.cpp src/script_classifier.cpp
OBJS = $(subst .cpp,.o,$(SRCS))

TARGET = script_classifier

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS) $(TARGET)
