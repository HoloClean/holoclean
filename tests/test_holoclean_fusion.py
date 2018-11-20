import unittest

import holoclean

class TestHolocleanFusion(unittest.TestCase):
    def test_flight(self):
        # 1. Setup a HoloClean session.
        hc = holoclean.HoloClean(pruning_topk=0.1, epochs=30, weight_decay=0.01, threads=20, batch_size=1, verbose=True, timeout=3*60000).session

        # 2. Load training data and denial constraints.
        hc.load_data('flight', '../testdata/flight.csv', entity_col='flight', src_col='src')

if __name__ == '__main__':
    unitttest.main()
