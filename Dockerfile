FROM waleedka/modern-deep-learning

RUN mkdir /code
WORKDIR /code

COPY requirements.txt /code/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -U scipy