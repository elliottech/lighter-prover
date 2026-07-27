# Bench

Benchmarks the full block proving pipeline (pre-execution, heavy/light tx proofs, chain recursion, and the final block proof) against a block witness, reporting per-proof and total proving times.

## Build

```sh
make build
```

Compiles the `bench` binary in release mode and places it in this directory.

## Run

```sh
make run
```

Runs `./bench` against `bench_test.json` with the default configuration. Variables can be overridden:

```sh
make run TX_COUNT=500 HEAVY_TX_PER_PROOF=4 LIGHT_TX_PER_PROOF=10 WITNESS=bench_test.json
```

| Variable | Default | Description |
| --- | --- | --- |
| `TX_COUNT` | `500` | Number of txs the block is filled with. |
| `HEAVY_TX_PER_PROOF` | `4` | Number of heavy txs proven per proof |
| `LIGHT_TX_PER_PROOF` | `10` | Number of light txs proven per proof |
| `WITNESS` | `bench_test.json` | Path to the block witness JSON file. Regular block witnesses work too; their txs are proven as-is and `TX_COUNT` is ignored. |

Or build and run in one step:

```sh
make build-and-run
```

The binary can also be invoked directly:

```sh
./bench --tx-count 500 --heavy-tx-per-proof 4 --light-tx-per-proof 10 --witness bench_test.json
```

Once built, the `bench` executable and the witness JSON can be copied across machines, placed at the same level, and run independently from the repository.
