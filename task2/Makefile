ID := 2022B4A70596P
FILE1 := openmp_sol1_$(ID).c
FILE2 := openmp_sol2_$(ID).c
FILE3 := openmp_sol3_$(ID).c

BIN1 := tg1
BIN2 := tg2
BIN3 := tg3

CC := gcc
CFLAGS := -O2 -Wall -Wextra -std=c11 -fopenmp
LDFLAGS := -fopenmp

.PHONY: all clean $(ID)
all: $(ID)

$(ID): $(BIN1) $(BIN2) $(BIN3)
	@echo "Build complete."
	@echo "Run using: ./$(ID) <num_threads>"
	@echo '#!/bin/bash' > $(ID)
	@echo 'THREADS=$$1' >> $(ID)
	@echo 'if [ -z "$$THREADS" ]; then echo "Usage: ./$(ID) <threads>"; exit; fi' >> $(ID)

	@echo 'echo "----FILE 1 : Basic Copy Version----"' >> $(ID)
	@echo 'for S in static dynamic guided; do' >> $(ID)
	@echo '  echo ""; echo "[FILE1] Schedule=$$S";' >> $(ID)
	@echo '  OMP_SCHEDULE=$$S ./$(BIN1) $$THREADS;' >> $(ID)
	@echo 'done' >> $(ID)

	@echo 'echo ""; echo "----FILE 2 : Pointer Swap Version----"' >> $(ID)
	@echo 'for S in static dynamic guided; do' >> $(ID)
	@echo '  echo ""; echo "[FILE2] Schedule=$$S";' >> $(ID)
	@echo '  OMP_SCHEDULE=$$S ./$(BIN2) $$THREADS;' >> $(ID)
	@echo 'done' >> $(ID)

	@echo 'echo ""; echo "----FILE 3 : Symmetry Version----"' >> $(ID)
	@echo 'for S in static dynamic guided; do' >> $(ID)
	@echo '  echo ""; echo "[FILE3] Schedule=$$S";' >> $(ID)
	@echo '  OMP_SCHEDULE=$$S ./$(BIN3) $$THREADS;' >> $(ID)
	@echo 'done' >> $(ID)

	@chmod +x $(ID)

$(BIN1): $(FILE1)
	$(CC) $(CFLAGS) $< -o $@

$(BIN2): $(FILE2)
	$(CC) $(CFLAGS) $< -o $@

$(BIN3): $(FILE3)
	$(CC) $(CFLAGS) $< -o $@
clean:
	rm -f $(BIN1) $(BIN2) $(BIN3) $(ID)