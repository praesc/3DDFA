#!/bin/bash

# Script to help run AIMET docker
# Modify based on user details

WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
docker_container_name=aimet-dev-last 
docker_image_name=aimet-dev-docker:last
port_id=8888

nvidia-docker run -p ${port_id}:${port_id} --rm -it -u $(id -u ${USER}):$(id -g ${USER}) --shm-size 8G  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro  -v ${WORKSPACE}:${WORKSPACE}   -v "/local/mnt/workspace":"/local/mnt/workspace" -v /home/${USER}/3DDFA:${WORKSPACE}/TDDFA --entrypoint /bin/bash -w ${WORKSPACE} --hostname ${docker_container_name} ${docker_image_name}
