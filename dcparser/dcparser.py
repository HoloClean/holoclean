import logging
import os
import time

from .constraint import DenialConstraint


class Parser:
    """
    This class creates interface for parsing denial constraints
    """

    def __init__(self, env, dataset):
        """
        Constructing parser interface object
        :param session: session object
        """
        self.env = env
        self.ds = dataset
        self.dc_strings = []
        self.dcs = {}

    def load_denial_constraints(self, f_path, f_name):
        """
        Loads denial constraints from line-separated txt file
        :param file_path: path to dc file
        :param all_current_dcs: list of current dcs in the session
        :return: list of Denial Constraint strings and their respective Denial Constraint Objects
        """
        tic = time.clock()
        if not self.ds.raw_data:
            status = 'NO dataset specified'
            toc = time.clock()
            return status, toc - tic
        attrs = self.ds.raw_data.get_attributes()
        try:
            dc_file = open(os.path.join(f_path,f_name), 'r')
            status = "OPENED constraints file successfully"
            logging.debug(status)
            for line in dc_file:
                if not line.isspace():
                    line = line.rstrip()
                    self.dc_strings.append(line)
                    self.dcs[line] = (DenialConstraint(line,attrs))
            status = 'DONE Loading DCs from ' + f_name
        except Exception:
            logging.error('loading constraints for file %s', f_name)
            raise
        toc = time.clock()
        return status, toc - tic

    def get_dcs(self):
        return self.dcs
