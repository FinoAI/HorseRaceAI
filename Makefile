CC = clang
CFLAGS = -O3 -Wall -Wextra -Iinclude
OBJCFLAGS = -O3 -Wall -Wextra -Iinclude -fobjc-arc
LDFLAGS = -framework Foundation -framework Metal

BUILD_DIR = build
BIN_DIR = bin
CLI_TARGET = $(BIN_DIR)/horse_race_metal_cli
BENCH_TARGET = $(BIN_DIR)/benchmark
ENS_TARGET = $(BIN_DIR)/ensemble_search

all: $(CLI_TARGET) $(BENCH_TARGET) $(ENS_TARGET)

$(CLI_TARGET): $(BUILD_DIR)/horse_race_metal.o $(BUILD_DIR)/main.o
	@mkdir -p $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@
	@echo "[Build] Successfully compiled $(CLI_TARGET)"

$(BENCH_TARGET): $(BUILD_DIR)/horse_race_metal.o $(BUILD_DIR)/benchmark.o
	@mkdir -p $(BIN_DIR)
	$(CC) $^ $(LDFLAGS) -o $@
	@echo "[Build] Successfully compiled $(BENCH_TARGET)"

$(ENS_TARGET): src/metal/ensemble_search.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@
	@echo "[Build] Successfully compiled $(ENS_TARGET)"

$(BUILD_DIR)/horse_race_metal.o: src/metal/horse_race_metal.m include/horse_race_metal.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/metal/main.c include/horse_race_metal.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/benchmark.o: src/metal/benchmark.c include/horse_race_metal.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean
