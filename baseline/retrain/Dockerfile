FROM levkuznetsov/gcsfuse-docker
FROM pytorch/pytorch:latest
USER root

WORKDIR /opt

ADD requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

ADD gcp-credentials.json /opt/gcp-credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/opt/gcp-credential.json"

ENV WANDB_API_KEY="4a3c06c226af9754cf91a2ff4a5616bd1eede964"

ADD train.py /opt/train.py
RUN chmod +x /opt/train.py

ENTRYPOINT ["python"]
CMD ["/opt/train.py"]

ENV NB_PREFIX /