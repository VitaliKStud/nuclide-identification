from sqlalchemy import create_engine

class DB:
    USER = "admin"
    PASSWORD = "admin"
    HOST = "localhost"
    PORT = "5432"
    DATABASE = "nuclide"
    ENGINE = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")

class PATH:
    MEASUREMENTS = "data\\measurements\\"
    NUCLIDES = "data\\nuclides\\"
    OUTPUT = "data\\"
