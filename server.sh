
MODEL_NAME='chinanews_albert'

docker run -t --rm -p 8500:8500  \
   -v "$(pwd)/serving/${MODEL_NAME}:/models/${MODEL_NAME}" \
   -e MODEL_NAME=${MODEL_NAME}  tensorflow/serving:1.14.0
