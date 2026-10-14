# Contributing to hana-models

Thank you for helping. This file says how to set up, what a change needs, and how we review it.

## Set up

```bash
git clone https://github.com/hanaml-lab/hana-models && cd hana-models
uv sync                                                    # Python 3.10+, the four packages and dev tools
cargo build --release --manifest-path packages/serve-bench/Cargo.toml
```

The Android app needs JDK 21 and an Android SDK (platform 36). The llama.cpp bridge needs CMake 3.22 or later.
Nothing else needs a GPU: the tests and the example run offline on a laptop.

## Check a change

```bash
make lint            # ruff, rustfmt, clippy
make test            # Rust tests, each Python package's tests, the integration test
make example         # the factory end to end
make test-android    # if you touched packages/small-lm/android
```

CI runs the same commands on every pull request.

## What a pull request needs

- **Tests through the public interface.** Write the failing test first, at the seam a caller uses (a command, a
  pipeline step, a module's entry point), then the code that makes it pass. Expected values come from a worked
  example or the spec, not from re-running the code.
- **One concern per pull request**, with a description of what changed and why.
- **A line in [CHANGELOG.md](CHANGELOG.md)** under the next version.
- **Numbers with their source.** A README or docstring that quotes a result says which command or which paper
  produced it.

## Commit messages

Conventional and lower case, as in the history: `type(scope): summary`, where the type is `feat`, `fix`, `test`,
`docs`, `ci`, `refactor` or `chore`, and the scope is the package (`distill-kit`, `small-lm`, `serve-bench`,
`hana-models`). For example: `feat(distill-kit): plan a GATE run before training`.

## Security

Do not open a public issue for a vulnerability. Follow the
[security policy](https://github.com/hanaml-lab/.github/blob/main/SECURITY.md).

## Licence

Contributions are accepted under the [Apache-2.0 licence](LICENSE) of this repository (inbound = outbound): by
opening a pull request you agree that your contribution is licensed under it.
