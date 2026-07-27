TX_COUNT ?= 30
HEAVY_TX_PER_PROOF ?= 4
LIGHT_TX_PER_PROOF ?= 10
WITNESS ?= bench_test.json

build:
	@echo ""
	@export RUSTFLAGS="-C target-cpu=native"
	cargo build --release --bin bench;
	@mv ../target/release/bench bench;

run: 
	@echo ""
	@./bench --tx-count $(TX_COUNT) --heavy-tx-per-proof $(HEAVY_TX_PER_PROOF) --light-tx-per-proof $(LIGHT_TX_PER_PROOF) --witness $(WITNESS)

build-and-run: build run
