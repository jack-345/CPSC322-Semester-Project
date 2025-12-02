from mysklearn import myutils
import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        col_vals = []
        if isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError(f"Column '{col_identifier}' does not exist")
            col_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError(f"Column index '{col_identifier}' is out of bounds")
            col_index = col_identifier
        else:
            raise ValueError(f"Column identifiers have to be string or int")
        
        for row in self.data:
            if col_index < len(row):
                val = row[col_index]
                if include_missing_values or val != 'NA':
                    col_vals.append(val)
        return col_vals

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                if val != 'NA':
                    try:
                        numeric_val = float(val)
                        self.data[i][j] = numeric_val
                    except (ValueError, TypeError):
                        pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        #sorting indexes
        sorted_indexes = sorted(row_indexes_to_drop, reverse = True)

        #dropping/deleting
        for index in sorted_indexes:
            if 0 <= index <= len(self.data):
                del self.data[index]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, 'r', newline = '', encoding = 'utf-8') as file:
            csv_reader = csv.reader(file)
            self.column_names = next(csv_reader)
            self.data = []
            for row in csv_reader:
                self.data.append(row)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(self.column_names)

            for row in self.data:
                csv_writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        key_indexes = []
        for col_name in key_column_names:
            if col_name in self.column_names:
                key_indexes.append(self.column_names.index(col_name))
            else:
                raise ValueError(f"Column '{col_name}' does not exist")
        
        viewed_keys = set()
        dupe_indexes = []

        for i, row in enumerate(self.data):
            key = tuple(row[j] if j < len(row) else None for j in key_indexes)
            if key in viewed_keys:
                dupe_indexes.append(i)
            else:
                viewed_keys.add(key)
        return dupe_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        remove_rows = []
        for i, row in enumerate(self.data):
            if 'NA' in row:
                remove_rows.append(i)
        self.drop_rows(remove_rows)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)

        num_vals = []
        for row in self.data:
            if col_index < len(row) and row[col_index] != 'NA':
                try:
                    num_vals.append(float(row[col_index]))
                except (ValueError, TypeError):
                    pass
        
        if not num_vals:
            return
        
        avg = sum(num_vals)/len(num_vals)

        for row in self.data:
            if col_index < len(row) and row[col_index] == 'NA':
                row[col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stat_table = MyPyTable(["attribute", "min", "max", "mid", "avg", "median"])

        for col_name in col_names:
            if col_name not in self.column_names:
                continue
            num_vals = []
            col_index = self.column_names.index(col_name)

            for row in self.data:
                if col_index < len(row) and row[col_index] != 'NA':
                    try:
                        num_vals.append(float(row[col_index]))
                    except (ValueError, TypeError):
                        pass

            if not num_vals:
                continue

            # min, max, mid, avg (mean)
            num_vals.sort()
            min_val = min(num_vals)
            max_val = max(num_vals)
            mid_val = (min_val + max_val)/2
            avg_val = sum(num_vals)/len(num_vals)

            # median
            n = len(num_vals)
            if (n % 2) == 0:
                median_val = (num_vals[n//2 - 1] + num_vals[n//2])/2
            else:
                median_val = num_vals[n//2]
            stat_table.data.append([col_name, min_val, max_val, mid_val, avg_val, median_val])

        return stat_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_indexes = []
        other_key_indexes = []

        for col_name in key_column_names:
            if col_name in self.column_names:
                self_key_indexes.append(self.column_names.index(col_name))
            else:
                raise ValueError(f"Column '{col_name}' does not exist in first table")
            
            if col_name in other_table.column_names:
                other_key_indexes.append(other_table.column_names.index(col_name))
            else:
                raise ValueError(f"Column '{col_name}' does not exist in second table")
            
            new_col_names = copy.deepcopy(self.column_names)
            
            for i, col_name in enumerate(other_table.column_names):
                if col_name not in key_column_names:
                    new_col_names.append(col_name)
        join_data = []

        for self_row in self.data:
            self_key = tuple(self_row[i] if i < len(self_row) else None for i in self_key_indexes)

            for other_row in other_table.data:
                other_key = tuple(other_row[i] if i < len(other_row) else None for i in other_key_indexes)
                if self_key == other_key:
                    join_row = copy.deepcopy(self_row)

                    for i, col_name in enumerate(other_table.column_names):
                        if col_name not in key_column_names:
                            if i < len(other_row):
                                join_row.append(other_row[i])
                            else:
                                join_row.append('NA')
                    join_data.append(join_row)
        return MyPyTable(new_col_names, join_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        # key col indexes for both tables
        self_key_indexes = []
        other_key_indexes = []

        for col_name in key_column_names:
            if col_name in self.column_names:
                self_key_indexes.append(self.column_names.index(col_name))
            else:
                raise ValueError(f"Column '{col_name}' does not exist in 1st table")
            
            if col_name in other_table.column_names:
                other_key_indexes.append(other_table.column_names.index(col_name))
            else:
                raise ValueError(f"Column '{col_name}' does not exist in 2nd table")
            
        # creating new col names
        new_col_names = copy.deepcopy(self.column_names)
        for col_name in other_table.column_names:
            if col_name not in key_column_names:
                new_col_names.append(col_name)
        
        join_data = []
        other_match_rows = set()

        # rows from self
        for self_row in self.data:
            self_key = tuple(self_row[i] if i < len(self_row) else None for i in self_key_indexes)
            matched = False

            # looking for matching rows between other table
            for j, other_row in enumerate(other_table.data):
                other_key = tuple(other_row[i] if i < len(other_row) else None for i in other_key_indexes)
                if self_key == other_key:
                    matched = True
                    other_match_rows.add(j)
                    join_row = copy.deepcopy(self_row)

                    # non-key cols from other
                    for i, col_name in enumerate(other_table.column_names):
                        if col_name not in key_column_names:
                            if i < len(other_row):
                                join_row.append(other_row[i])
                            else:
                                join_row.append('NA')
                    join_data.append(join_row)

            # self row with NA for other table cols
            if not matched:
                join_row = copy.deepcopy(self_row)
                # NAs
                for col_name in other_table.column_names:
                    if col_name not in key_column_names:
                        join_row.append('NA')
                join_data.append(join_row)
        
        # rows from other that are unmatched
        for j, other_row in enumerate(other_table.data):
            if j not in other_match_rows:
                join_row = []

                for col_name in self.column_names:
                    if col_name in key_column_names:
                        key_index = key_column_names.index(col_name)
                        other_col_index = other_table.column_names.index(col_name)

                        if other_col_index < len(other_row):
                            join_row.append(other_row[other_col_index])
                        else:
                            join_row.append('NA')
                    else:
                        join_row.append('NA')
                
                for i, col_name in enumerate(other_table.column_names):
                    if col_name not in key_column_names:
                        if i < len(other_row):
                            join_row.append(other_row[i])
                        else:
                            join_row.append('NA')
                join_data.append(join_row)
        return MyPyTable(new_col_names, join_data)