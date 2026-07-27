# Compile
# rust:1.91 — rust-toolchain.toml pins `channel = "1.91.1"`. On the 1.89 image
# the cargo step died in 13 seconds, far too fast for a Hanzo Index release
# build: rustup has to fetch the pinned toolchain before compiling and that is
# where it stops. Third instance of this shape today — golang:1.23 vs go 1.26.4
# in hanzoai/search-fts5, node:20 vs pnpm@11 in hanzoai/world. The base image
# must satisfy the toolchain the repo declares.
FROM    rust:1.91-alpine3.22 AS compiler

RUN     apk add -q --no-cache build-base openssl-dev

WORKDIR /

ARG     COMMIT_SHA
ARG     COMMIT_DATE
ARG     GIT_TAG
ARG     EXTRA_ARGS
ENV     VERGEN_GIT_SHA=${COMMIT_SHA} VERGEN_GIT_COMMIT_TIMESTAMP=${COMMIT_DATE} VERGEN_GIT_DESCRIBE=${GIT_TAG}
ENV     RUSTFLAGS="-C target-feature=-crt-static"

COPY    . .
RUN     set -eux; \
        apkArch="$(apk --print-arch)"; \
        # ${EXTRA_ARGS:-} — `set -eux` turns on `set -u`, and EXTRA_ARGS is an ARG
        # with no default, so an unpassed build-arg aborted the shell with
        #   /bin/sh: EXTRA_ARGS: parameter not set
        # before cargo ran at all.
        cargo build --release -p search -p searchtool ${EXTRA_ARGS:-}

# Run
FROM    ghcr.io/hanzoai/alpine:3.22
LABEL   org.opencontainers.image.source="https://github.com/hanzoai/index"
LABEL   org.opencontainers.image.title="Hanzo Index"
LABEL   org.opencontainers.image.description="Full-text and vector index: documents in, ranked results out"
LABEL   org.opencontainers.image.vendor="Hanzo AI Inc."
LABEL   org.opencontainers.image.url="https://hanzo.ai"

ENV     INDEX_HTTP_ADDR 0.0.0.0:7700
ENV     INDEX_SERVER_PROVIDER docker

RUN     apk add -q --no-cache libgcc tini curl

# The crates are `search` and `searchtool` (crates/*/Cargo.toml, no [[bin]]
# override), so cargo emits binaries by those names. The upstream symlink for
# pre-v0.27.0 containers is dropped — we have no such containers.
COPY    --from=compiler /target/release/search /bin/search
COPY    --from=compiler /target/release/searchtool /bin/searchtool

# The database lives here and nowhere else, so it is both the WORKDIR and the
# mount point a deployment is expected to attach a volume to. Unset
# INDEX_DB_PATH resolves relative to it.
WORKDIR /data


EXPOSE  7700/tcp

ENTRYPOINT ["tini", "--"]
CMD     /bin/search
