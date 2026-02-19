ID := 2022xxxx
SRC := custom_lock.c
TARGET := $(ID)

CC := gcc
CFLAGS  := -Wall -Wextra -pedantic -std=c11 -pthread -O2
LDFLAGS := -pthread

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling $(SRC)..."
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)
	@echo "Build complete"

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
