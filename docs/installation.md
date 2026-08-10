# Installation

## As a library dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
trx-rs = { git = "https://github.com/tee-ar-ex/trx-rs" }
```

## Local development

```bash
git clone https://github.com/tee-ar-ex/trx-rs
cd trx-rs
cargo build --release
```

## Command-line tool

The `trxrs` binary ships with the crate under `src/bin/trxrs.rs`.

Build and install it:

```bash
cargo install --path .
trxrs --help
```

## Testing

```bash
# Full test suite
cargo test --workspace

# Formatting check
cargo fmt --check

# Lint with warnings denied
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Coverage report (requires cargo-llvm-cov)
cargo llvm-cov --workspace --html
```

## Python bindings

Python bindings live in the `python/` directory. See the [trxrs Python docs](https://trxrs.readthedocs.io) for details.
