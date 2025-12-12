FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /app
COPY main.py main.py

CMD [ "python", "./main.py", "--bokeh" ]
