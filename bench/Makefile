
build:
	@echo ""
	@export RUSTFLAGS="-C target-cpu=native"
	cargo build --release --bin bench;
	@mv ../target/release/bench bench;

run: 
	@echo ""
	@./bench

build-and-run: build run
