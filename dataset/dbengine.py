import logging
import sqlalchemy as sql
import time
from string import Template
from multiprocessing import Pool
from functools import partial
import psycopg2

index_template = Template('CREATE INDEX $idx_title ON $table ($attr)')
drop_table_template = Template('DROP TABLE IF EXISTS $tab_name')
create_table_template = Template('CREATE TABLE $tab_name AS ($stmt)')

def execute_query(args, conn_args):
    query_id = args[0]
    query = args[1]
    logging.debug("Starting to execute query %s with id %s", query, query_id)
    tic = time.clock()
    con = psycopg2.connect(conn_args)
    cur = con.cursor()
    cur.execute(query)
    res = cur.fetchall()
    con.close()
    toc = time.clock()
    logging.debug('Time to execute query with id %d: %.2f secs' % (query_id, (toc - tic)))
    return res

def execute_query_w_backup(args, conn_args, timeout):
    query_id = args[0]
    query = args[1][0]
    query_backup = args[1][1]
    logging.debug("Starting to execute query %s with id %s", query, query_id)
    tic = time.clock()
    con = psycopg2.connect(conn_args)
    cur = con.cursor()
    cur.execute("SET statement_timeout to %d;"%timeout)
    try:
        cur.execute(query)
        res = cur.fetchall()
    except psycopg2.extensions.QueryCanceledError as e:
        logging.debug("Failed to execute query %s with id %s. Timeout reached.", query, query_id)
        logging.debug("Starting to execute backup query %s with id %s", query_backup, query_id)
        con.close()
        con = psycopg2.connect(conn_args)
        cur = con.cursor()
        cur.execute(query_backup)
        res = cur.fetchall()
        if len(res) == 1:
            logging.info(res)
        con.close()
    toc = time.clock()
    logging.debug('Time to execute query with id %d: %.2f secs', query_id, toc - tic)
    return res

class DBengine:
    def __init__(self, user, pwd, db, host='localhost', port=5432, pool_size=20, timeout=60000):
        self.POOL_MAX = pool_size
        self.timeout = timeout
        self.pool = Pool(self.POOL_MAX)
        url = 'postgresql+psycopg2://{}:{}@{}:{}/{}'
        url = url.format(user, pwd, host, port, db)
        self.conn = url
        con = 'dbname={} user={} password={} host={} port={}'
        con = con.format(db, user, pwd, host, port)
        self.conn_args = con
        self.engine = sql.create_engine(url, client_encoding='utf8', pool_size=pool_size)

    # Executes queries in parallel.
    def execute_queries(self, queries):
        logging.debug('Preparing to execute %d queries.', len(queries))
        tic = time.clock()
        # TODO(python3): Modify pool to context manager (with statement)
        results = self.pool.map(partial(execute_query, conn_args=self.conn_args), [(idx, q) for idx, q in enumerate(queries)])
        toc = time.clock()
        logging.debug('Time to execute %d queries: %.2f secs', len(queries), toc-tic)
        return results

    # Executes queries that have backups in parallel. Used in featurization.
    def execute_queries_w_backup(self, queries):
        logging.debug('Preparing to execute %d queries.', len(queries))
        tic = time.clock()
        # TODO(python3): Modify pool to context manager (with statement)
        results = self.pool.map(
            partial(execute_query_w_backup, conn_args=self.conn_args, timeout=self.timeout),
            [(idx, q) for idx, q in enumerate(queries)])
        toc = time.clock()
        logging.debug('Time to execute %d queries: %.2f secs', len(queries), toc-tic)
        return results

    # Executes a single query using current connection.
    def execute_query(self, query):
        tic = time.clock()
        conn = self.engine.connect()
        result = conn.execute(query).fetchall()
        conn.close()
        toc = time.clock()
        logging.debug('Time to execute query: %.2f secs', toc-tic)
        return result

    def create_db_table_from_query(self, name, query):
        tic = time.clock()
        drop = drop_table_template.substitute(tab_name=name)
        create = create_table_template.substitute(tab_name=name, stmt=query)
        conn = self.engine.connect()
        dropped = conn.execute(drop)
        created = conn.execute(create)
        conn.close()
        toc = time.clock()
        logging.debug('Time to create table: %.2f secs', toc-tic)
        return True

    def create_db_index(self, name, table, attr_list):
        stmt = index_template.substitute(idx_title=name, table=table, attr=','.join(attr_list))
        tic = time.clock()
        conn = self.engine.connect()
        result = conn.execute(stmt)
        conn.close()
        toc = time.clock()
        logging.debug('Time to create index: %.2f secs', toc-tic)
        return result
