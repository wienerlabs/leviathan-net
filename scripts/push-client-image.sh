# Script for pushing the client image to Docker Hub during the CI pipeline.
# Uses skopeo to stream the image directly to the registry without loading
# into Docker's local storage, avoiding double disk usage.
set -eu

# An unset repository variable arrives here as an empty string, which would push
# to "docker://:latest" and fail with something that does not name the cause.
: "${DOCKER_REPOSITORY:?DOCKERHUB_REPO variable is not set on this repository or org}"
: "${REGISTRY_USERNAME:?DOCKERHUB_USERNAME variable is not set on this repository or org}"
: "${REGISTRY_PASSWORD:?DOCKER_DEPLOY_KEY secret is not set on this repository or org}"

IMAGE_PATH=$(cat image-path.txt)
"$IMAGE_PATH" | skopeo copy \
    --dest-creds "${REGISTRY_USERNAME}:${REGISTRY_PASSWORD}" \
    docker-archive:/dev/stdin \
    "docker://${DOCKER_REPOSITORY}:latest"
