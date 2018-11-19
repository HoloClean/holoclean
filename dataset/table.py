import os
import pandas as pd
from enum import Enum

class Source(Enum):
    FILE = 1
    DF   = 2
    DB   = 3
    SQL  = 4

class Table:
    """
    A wrapper class for Dataset Tables.
    """
    def __init__(self, name, src, *args):
        self.name = name
        self.index_count = 0
        self.df = pd.DataFrame()
        if src == Source.FILE:
            if len(args) < 2:
                raise Exception("ERROR while loading table. File path and file name expected.Please provide <file_path> and <file_name>.")
            else:
                file_path = args[0]
                file_name = args[1]
                if len(args) == 3:
                    na_values = args[2]
                else:
                    na_values = None
                self.df = pd.read_csv(os.path.join(file_path,file_name), dtype=str, na_values=na_values)
                print("DEBUGGING: {}".format(self.df.loc[1]))
                # Normalize to lower strings and strip whitespaces.
                # TODO: No support for numerical values. To be added.
                for attr in self.df.columns.values:
                    if attr != '_tid_':
                        self.df[attr] = self.df[attr].apply(lambda x: x.lower().strip() if type(x) == str else x)
                self.df.columns = map(str.lower, self.df.columns)
        elif src == Source.DF:
            if len(args) != 1:
                raise Exception("ERROR while loading table. Dataframe expected. Please provide <dataframe>.")
            else:
                self.df = args[0]
        elif src == Source.DB:
            if len(args) != 1:
                raise Exception("ERROR while loading table. DB connection expected. Please provide <db_conn>")
            else:
                db_conn = args[0]
                self.df = pd.read_sql_table(name, db_conn)
        elif src == Source.SQL:
            if len(args) != 2:
                raise Exception("ERROR while loading table. SQL Query and DB engine expected. Please provide <query> and <db_engine>.")
            else:
                tab_query = args[0]
                dbengine = args[1]
                dbengine.create_db_table_from_query(self.name, tab_query)
                self.df = pd.read_sql_table(name, dbengine.conn)

    def store_to_db(self, con, if_exists='replace', index=False, index_label=None):
        # TODO: This version supports single session, single worker.
        self.df.to_sql(self.name, con, if_exists=if_exists, index=index, index_label=index_label)

    def get_attributes(self):
        if not self.df.empty:
            return list(self.df.columns.values)
        else:
            raise Exception("Empty Dataframe associated with table "+self.name+". Cannot return attributes.")

    def create_df_index(self, attr_list):
        self.df.set_index(attr_list, inplace=True)

    def create_db_index(self, dbengine, attr_list):
        index_name = self.name+'_'+str(self.index_count)
        try:
            dbengine.create_db_index(index_name, self.name, attr_list)
            self.index_count += 1
        except:
            raise Exception("ERROR while creating index for table %s on attributes %s"%(self.name, str(attr_list)))
        return
