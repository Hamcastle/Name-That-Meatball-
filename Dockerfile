FROM ubuntu:16.04

# Install Python.
RUN \
    apt-get update && \
    apt-get install -y python3 python3-dev python3-pip python3-virtualenv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR app

# reqs from file, to speed up dev iteration
RUN pip3 install Werkzeug Flask numpy Keras gevent pillow h5py tensorflow

COPY . .

RUN pip3 install -r requirements.txt
##
CMD [ "python3" , "app.py"]