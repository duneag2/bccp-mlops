FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install pandas psycopg2-binary

ARG DATASET
ARG TARGET_DAY

ENV DATASET ${DATASET}

COPY data_generator_${TARGET_DAY}.py data_generator.py
COPY ${DATASET}_monday.json ${DATASET}_monday.json
COPY ${DATASET}_tuesday.json ${DATASET}_tuesday.json

ENTRYPOINT ["python", "data_generator.py", "--db-host"]
# Change CMD to solve host finding error
CMD ["localhost"]