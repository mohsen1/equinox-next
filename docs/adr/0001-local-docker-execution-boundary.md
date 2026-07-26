# ADR-0001: Keep Docker control outside candidate sessions

Status: accepted for local contract proof
Date: 2026-07-26

## Decision

The execution service runs as a dedicated Docker Compose service and owns isolated
logical session directories. Candidate work runs only through registered CAD workload
profiles inside that container boundary. The local stack does not mount the host Docker
socket and does not expose a Docker API to candidate or policy code.

The obtained snapshot fidelity is `logical_restore`: canonical state JSON plus a
deterministic digest and fidelity probe. It is not process-memory, filesystem, microVM, or
full-runtime fidelity.

## Why

The local milestone must prove state, fork, fencing, recovery, and evidence contracts on
Docker without claiming production isolation. Docker documents that daemon control can
mount and modify the host filesystem and is effectively privileged. A raw host socket
inside an application service would therefore turn an execution bug into host control.

The selected boundary is smaller and safer for deterministic fixture CAD actions. It
keeps runtime-cursor identities and files isolated while postponing per-session containers
or microVMs until measurements justify them.

## Rejected options

- Raw `/var/run/docker.sock` mount: simple, but explicitly unsafe and unnecessary for
  local fixture execution.
- Docker-in-Docker: adds a privileged daemon, image lifecycle, and recovery surface
  without improving the contract proof.
- Firecracker on macOS: not a directly supported local host path and would overstate
  production fidelity.

## Consequences

- Local acceptance proves Compose-backed process isolation and logical restore only.
- Candidate profiles have no network, credentials, hidden evidence, shell input, or
  arbitrary commands.
- Production isolation remains an open measured spike before CAD research beta.
- If per-session container compatibility becomes a beta requirement, replace the provider
  behind the runtime interface and rerun fidelity and recovery conformance.

Evidence: [Docker Engine daemon attack surface](https://docs.docker.com/engine/security/)
and [Docker rootless mode](https://docs.docker.com/engine/security/rootless/).
