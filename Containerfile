FROM docker.io/mambaorg/micromamba:alpine

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml

RUN micromamba install -y -n base -f /tmp/env.yaml && \
  micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY . /app

CMD ["python", "main.py"]

