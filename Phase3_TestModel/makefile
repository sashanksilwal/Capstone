OBJS = main.o  
CC = c++
CFLAGS = -std=c++17 -Iinclude
LIBS =  lib/libonnxruntime.dylib
PROGRAM_NAME = main

all: $(PROGRAM_NAME)
$(PROGRAM_NAME): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LIBS)
main.o: main.cpp
	$(CC) $(CFLAGS) -c $< -o $@ 
clean:
	rm -f $(OBJS) $(PROGRAM_NAME)
 