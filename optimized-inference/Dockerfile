FROM python:3.10.4-slim-buster

WORKDIR usr/src/app

# set environment variables
# Prevents Python from writing pyc files to disc
# Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1 
ENV PYTHONUNBUFFERED 1

COPY requirements.txt usr/src/app/requirements.txt
# install system dependencies
RUN apt-get update \
  && apt-get -y install netcat gcc\
  && apt-get clean

RUN python -m pip install --upgrade pip && pip install -r usr/src/app/requirements.txt
RUN python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable

# copy content to app
COPY . /usr/src/app

EXPOSE 8080
#ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 main:app -k uvicorn.workers.UvicornWorker --timeout 1000"
#CMD gunicorn --bind=0.0.0.0:8000 --workers 1 -k uvicorn.workers.UvicornWorker main:app 
# ENTRYPOINT exec gunicorn main:app -k uvicorn.workers.UvicornWorker
# ENTRYPOINT exec uvicorn main:app --workers 1 --reload --host 0.0.0.0 --port 8000
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]