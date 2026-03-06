USER=hdliu

nvidia-docker run --rm -it \
    -v /raid/${USER}_raid:/raid/${USER}_raid \
    -v /home/${USER}:/home/${USER} \
    --shm-size 8G \
    --gpus='"device=0,1,2,4"' \
    -p 8083:8083 \
    -p 7863:7863 \
    -p 10003:10003 \
    llava-env:5.0 bash