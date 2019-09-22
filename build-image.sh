version=`cat ./VERSION`
echo "version: $version"
docker build . -t "faceml"
docker tag faceml:latest samirsdoshi/faceml:latest 
docker tag faceml:latest samirsdoshi/faceml:$version
