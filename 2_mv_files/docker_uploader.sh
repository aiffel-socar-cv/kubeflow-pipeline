docker build -t mv_files .

docker tag mv_files:latest tseo/mv_files:0.6

docker push tseo/mv_files:0.6