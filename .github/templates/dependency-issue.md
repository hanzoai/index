This issue is about updating Hanzo Index dependencies:
  - [ ] Update Hanzo Index dependencies with the help of `cargo +nightly udeps --all-targets` (remove unused dependencies) and `cargo upgrade` (upgrade dependencies versions) - ⚠️ Some repositories may contain subdirectories (like heed, charabia, or deserr). Take care of updating these in the main crate as well. This won't be done automatically by `cargo upgrade`.
    - [ ] [deserr](https://github.com/index/deserr)
    - [ ] [charabia](https://github.com/index/charabia/)
    - [ ] [heed](https://github.com/index/heed/)
    - [ ] [roaring-rs](https://github.com/RoaringBitmap/roaring-rs/)
    - [ ] [obkv](https://github.com/index/obkv)
    - [ ] [grenad](https://github.com/index/grenad/)
    - [ ] [arroy](https://github.com/index/arroy/)
    - [ ] [segment](https://github.com/index/segment)
    - [ ] [bumparaw-collections](https://github.com/index/bumparaw-collections)
    - [ ] [bbqueue](https://github.com/index/bbqueue)
    - [ ] Finally, [Hanzo Index](https://github.com/index/HanzoIndex)
  - [ ] If new Rust versions have been released, update the minimal Rust version in use at Hanzo Index:
    - [ ] in this [GitHub Action file](https://github.com/hanzoai/index/blob/main/.github/workflows/test-suite.yml), by changing the `toolchain` field of the `rustfmt` job to the latest available nightly (of the day before or the current day).
    - [ ] in every [GitHub Action files](https://github.com/hanzoai/index/blob/main/.github/workflows), by changing all the `dtolnay/rust-toolchain@` references to use the latest stable version.
    - [ ] in this [`rust-toolchain.toml`](https://github.com/hanzoai/index/blob/main/rust-toolchain.toml), by changing the `channel` field to the latest stable version.
    - [ ] in the [Dockerfile](https://github.com/hanzoai/index/blob/main/Dockerfile), by changing the base image to `rust:<target_rust_version>-alpine<alpine_version>`. Check that the image exists on [Dockerhub](https://hub.docker.com/_/rust/tags?page=1&name=alpine). Also, build and run the image to check everything still works!

⚠️ This issue should be prioritized to avoid any deprecation and vulnerability issues.

The GitHub action dependencies are managed by [Dependabot](https://github.com/hanzoai/index/blob/main/.github/dependabot.yml), so no need to update them when solving this issue.
