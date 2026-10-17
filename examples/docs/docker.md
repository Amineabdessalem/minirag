# Docker

Docker packages an application and its dependencies into an image built from a Dockerfile. Each instruction creates a layer, and layers are cached so unchanged steps are not rebuilt.

Multi-stage builds compile code in one stage and copy only the final artefacts into a small runtime image, which reduces image size and attack surface. Containers are isolated with Linux namespaces and resource limits are enforced with cgroups.
