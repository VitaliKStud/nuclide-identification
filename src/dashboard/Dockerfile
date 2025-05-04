FROM python:3

WORKDIR /usr/src/app

COPY start_dashboard.py .
COPY pyproject.toml .
COPY config.py .

RUN pip install dash pandas plotly dash-table sqlalchemy psycopg2

CMD [ "python", "start_dashboard.py" ]