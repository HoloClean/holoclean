import csv
import pytest
from tempfile import NamedTemporaryFile

from detect.errorloaderdetector import ErrorsLoaderDetector


def test_errors_loader_valid_csv_file():
    tmp_file = NamedTemporaryFile(delete=False)
    with open(tmp_file.name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['_tid_', 'attribute'])  # Header.
        csv_writer.writerow([1, 'attr1'])
        csv_writer.writerow([1, 'attr2'])
        csv_writer.writerow([2, 'attr1'])
        csv_writer.writerow([3, 'attr2'])
    errors_loader_detector = ErrorsLoaderDetector(fpath=tmp_file.name)
    errors_df = errors_loader_detector.errors_df

    assert errors_df is not None
    assert errors_df.columns.tolist() == ['_tid_', 'attribute']
    assert len(errors_df) == 4


def test_errors_loader_invalid_csv_file():
    tmp_file = NamedTemporaryFile(delete=False)
    with open(tmp_file.name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['_tid_', 'invalid_column'])  # Header.
        csv_writer.writerow([1, 'val1'])

    with pytest.raises(Exception) as invalid_file_error:
        errors_loader_detector = ErrorsLoaderDetector(fpath=tmp_file.name)

    assert 'Invalid input file for ErrorsLoaderDetector' in str(invalid_file_error.value)
