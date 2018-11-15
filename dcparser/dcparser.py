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

    def load_denial_constraints(self, fpath):
        """
        Loads denial constraints from line-separated TXT file
        :param fpath: filepath to TXT file containing denial constraints
        """
        tic = time.clock()
        if not self.ds.raw_data:
            status = 'NO dataset specified'
            toc = time.clock()
            return status, toc - tic
        attrs = self.ds.raw_data.get_attributes()
        dc_file = open(fpath, 'r')
        status = "OPENED constraints file successfully"
        if self.env['verbose']:
            print (status)
        for line in dc_file:
            if not line.isspace():
                line = line.rstrip()
                self.dc_strings.append(line)
                self.dcs[line] = (DenialConstraint(line,attrs,self.env['verbose']))
        status = 'DONE Loading DCs from {fname}'.format(fname=os.path.basename(fpath))
        toc = time.clock()
        return status, toc - tic

    def get_dcs(self):
        return self.dcs
