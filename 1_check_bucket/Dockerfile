FROM gcr.io/google.com/cloudsdktool/cloud-sdk:alpine

# Install python/pip
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install google-cloud-storage

# GCP OAuth
ENV GOOGLE_APPLICATION_CREDENTIALS=/.gcp/aiffel-gn-3-c8c200820331.json
ENV GOOGLE_PROJECT_ID=aiffel-gn-3

RUN echo '[Credentials]\ngs_service_key_file = /.gcp/aiffel-gn-3-c8c200820331.json' \
    > /etc/boto.cfg
RUN mkdir /.gcp
COPY aiffel-gn-3-c8c200820331.json $GOOGLE_APPLICATION_CREDENTIALS

RUN gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS --project $GOOGLE_PROJECT_ID
RUN gcloud config set project $GOOGLE_PROJECT_ID

# Copy and Run python file
COPY check_bucket.py /
ENTRYPOINT [ "python", "check_bucket.py" ]
