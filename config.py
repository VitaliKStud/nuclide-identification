from sqlalchemy import create_engine


class DB:
    USER = "admin"
    PASSWORD = "admin"
    HOST = "localhost"  # TODO: FOR DOCKER DASHBOARD NEED TO BE EDITED AND SET AS DB (CONTAINERNAME)
    PORT = "5432"
    DATABASE = "nuclide"
    ENGINE = create_engine(
        f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    )


class MLFLOW:
    URI = "http://localhost:5000"


class PATH:
    MEASUREMENTS = "data\\measurements\\"
    NUCLIDES = "data\\nuclides\\"
    OUTPUT = "data\\"
