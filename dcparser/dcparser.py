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
        self.dcs = []

    def load_denial_constraints(self, fpath):
        """
        Loads denial constraints from line-separated TXT file
        
        :param fpath: filepath to TXT file containing denial constraints
        """
        tic = time.clock()
        if not self.ds.raw_data:
            status = 'No dataset specified'
            toc = time.clock()
            return status, toc - tic
        attrs = self.ds.raw_data.get_attributes()
        try:
            dc_file = open(fpath, 'r')
            status = "OPENED constraints file successfully"
            logging.debug(status)
            for line in dc_file:
                if not line.isspace():
                    line = line.rstrip()
                    self.dc_strings.append(line)
                    self.dcs.append(DenialConstraint(line, attrs))
            status = 'DONE Loading DCs from {fname}'.format(fname=os.path.basename(fpath))
        except Exception:
            logging.error('FAILED to load constraints from file %s', os.path.basename(fpath))
            raise
        toc = time.clock()
        return status, toc - tic

    def get_dcs(self):
        return self.dcs
