import pandas as pd
import time
from dataset import AuxTables


class DetectEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset

    def detect_errors(self, detectors):
        errors = []
        tic = time.clock()
        for detector in detectors:
            detector.setup(self.ds, self.env)
        for detector in detectors:
            tic = time.clock()
            error_df = detector.detect_noisy_cells()
            toc = time.clock()
            if self.env['verbose']:
                print("DONE with Error Detector: %s in %.2f secs"%(detector.name, toc-tic))
            errors.append(error_df)

        errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)
        errors_df['_cid_'] = errors_df.apply(lambda x: self.ds.get_cell_id(x['_tid_'], x['attribute']), axis=1)
        self.store_detected_errors(errors_df)
        status = "DONE with error detection."
        toc = time.clock()
        detect_time = toc - tic
        return status, detect_time

    def store_detected_errors(self, errors_df):
        if errors_df.empty:
            raise Exception("ERROR: Detected errors dataframe is empty.")
        self.ds.generate_aux_table(AuxTables.dk_cells, errors_df, store=True)
        self.ds.aux_tables[AuxTables.dk_cells].create_db_index(self.ds.engine, ['_cid_'])
