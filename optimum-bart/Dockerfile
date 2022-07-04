# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with source build for CPU

FROM python:3.10-slim-buster

WORKDIR /usr/src/app
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends bash git gcc

RUN python -m pip install -U setuptools && python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]

CMD ["/bin/bash", "-c python3 bart_optimum.py && tail -f /dev/null"]

# COPY requirements.txt usr/src/app/requirements.txt
# # install system dependencies
# RUN apt-get update \
#   && apt-get -y install netcat gcc\
#   && apt-get clean

# RUN python -m pip install --upgrade pip && pip install -r usr/src/app/requirements.txt
# RUN python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable

# # copy content to app
# COPY . /usr/src/app

# EXPOSE 8080
# #ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 main:app -k uvicorn.workers.UvicornWorker --timeout 1000"
# #CMD gunicorn --bind=0.0.0.0:8000 --workers 1 -k uvicorn.workers.UvicornWorker main:app 
# # ENTRYPOINT exec gunicorn main:app -k uvicorn.workers.UvicornWorker
# # ENTRYPOINT exec uvicorn main:app --workers 1 --reload --host 0.0.0.0 --port 8000
# CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
