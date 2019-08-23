CONTAINER_NAME=kerastfopencv
docker stop ${CONTAINER_NAME}
docker inspect --format "0" "${CONTAINER_NAME}" >/dev/null 2>&1 && docker rm -v -f "${CONTAINER_NAME}"
docker run -m=12g  -dit \
  --name ${CONTAINER_NAME} \
  -p 8886:8888 \
  -p 6005:6006 \
  --volume "/Users/sdoshi/Development/keras/workspace":"/ML/keras" \
  --volume "/samir/ML":"/ML/MicrosoftML" \
  --volume "/samir/data":"/ML/images" \
  --workdir "/ML" \
  -e GRANT_SUDO=yes \
  --user root \
  -e DISPLAY=10.20.53.76:0 \
  ${CONTAINER_NAME}
sleep 10
docker logs ${CONTAINER_NAME}

#  --network global \
#  --link my-mysql_1:mysql \
