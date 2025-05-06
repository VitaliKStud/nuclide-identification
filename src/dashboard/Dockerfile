FROM python:3

WORKDIR /usr/src/app

COPY start_dashboard.py .
COPY pyproject.toml .

RUN pip install dash pandas plotly dash-table sqlalchemy psycopg2 PyYAML

CMD [ "python", "start_dashboard.py" ]