import random

from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def random_database():
    """
    Creates a random database in the testing Postgres instance and returns the
    name of the database.
    """
    # Setup connection with default credentials for testing.
    with connect(dbname='holo', user='holocleanuser', password='abcd1234', host='localhost') as conn:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            while True:
                # Generate a random DB name that is not already in Postgres.
                db_name = 'test_holo_{}'.format(random.randint(0, 1e6))
                cur.execute("""
                    SELECT EXISTS(
                        SELECT datname FROM pg_catalog.pg_database
                        WHERE datname = '{db_name}'
                    );
                """.format(db_name=db_name))
                if cur.fetchall()[0][0]:
                    continue

                cur.execute("CREATE DATABASE {db_name}".format(db_name=db_name))
                return db_name

def delete_database(db_name):
    with connect(dbname='holo', user='holocleanuser', password='abcd1234', host='localhost') as conn:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            # Terminate un-closed connections.
            cur.execute("""
            SELECT pid, pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{db_name}' AND pid <> pg_backend_pid();""".format(db_name=db_name))
            # Drop the database.
            cur.execute("DROP DATABASE IF EXISTS {db_name}".format(db_name=db_name))
