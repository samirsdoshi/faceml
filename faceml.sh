IMAGE_NAME=faceml
CONTAINER_NAME=faceml
docker stop ${CONTAINER_NAME}
docker inspect --format "0" "${CONTAINER_NAME}" >/dev/null 2>&1 && docker rm -v -f "${CONTAINER_NAME}"
#change volume path below 
docker run -m=14g  -dit \
  --name ${CONTAINER_NAME} \
  -p 8888:8888 \
  -p 6006:6006 \
  --volume "/samir/data":"/images" \
  --workdir "/faceml" \
  -e GRANT_SUDO=yes \
  --user root \
  ${IMAGE_NAME}
sleep 10
docker logs ${CONTAINER_NAME}