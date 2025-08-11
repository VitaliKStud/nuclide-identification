import yaml
import os
from sqlalchemy import create_engine


def load_config(path=None):
    """
    Loading config. A variable named 'CONFIG_FILE' (path to config-filge) should be already created
    in the environment.

    :param path:
    :type path:
    :return:
    :rtype:
    """
    config_file = path or os.environ.get("CONFIG_FILE")
    with open(f"config/{config_file}", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_engine():
    """
    Loading the PostgreSQL engine and creating the engine

    :return: PostgreSQL engine instance
    """
    username = load_config()["postges_db"]["username"]
    password = load_config()["postges_db"]["password"]
    host = load_config()["postges_db"]["host"]
    port = load_config()["postges_db"]["port"]
    database = load_config()["postges_db"]["database"]
    return create_engine(
        f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    )
