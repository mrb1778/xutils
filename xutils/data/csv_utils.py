import csv


def get_data(filename, skip_header=True):
    with open(filename, 'r') as csv_file:
        csv_file_reader = csv.reader(csv_file)
        if skip_header:
            next(csv_file_reader)
        return csv_file_reader


