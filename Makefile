.PHONY: install build-rust test test-python test-rust test-android lint example

install:
	uv sync

build-rust:
	cargo build --release --manifest-path packages/serve-bench/Cargo.toml

test: test-rust test-python

test-python: build-rust
	cd packages/distill-kit && uv run pytest -q
	cd packages/small-lm && uv run pytest -q
	cd packages/serve-bench/python && uv run pytest -q
	uv run pytest -q tests/integration

test-rust:
	cargo test --manifest-path packages/serve-bench/Cargo.toml

# The Kotlin router's unit tests; needs an Android SDK (ANDROID_HOME).
test-android:
	cd packages/small-lm/android && ./gradlew testDebugUnitTest

lint:
	uv run ruff check .
	cd packages/serve-bench && cargo fmt --all --check
	cargo clippy --manifest-path packages/serve-bench/Cargo.toml --all-targets -- -D warnings

example: build-rust
	uv run python examples/end_to_end.py
