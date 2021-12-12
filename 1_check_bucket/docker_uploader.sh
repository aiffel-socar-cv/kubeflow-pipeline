docker build -t check_bucket .

docker tag check_bucket:latest tseo/check_bucket:0.3

docker push tseo/check_bucket:0.3