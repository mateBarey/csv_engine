# engine/core.py

import csv
import sys
from itertools import islice, chain
from functools import reduce
from .constants import CMP_OPS, ARITH_OPS, LOGICAL_OPS
from .utils import SafeBinder


class ParseCSV:
    def __init__(self, file_path, has_headers=True, chunk_threshold=1000):
        self.data = file_path
        self.has_headers = has_headers
        self.chunk_threshold = chunk_threshold
        self._headers = []
        self._sampling_cache = []
        self._went_lazy = False

    def get_rows(self):
        """Load CSV rows into memory lazily, respecting header settings."""
        raw = self._raw_rows()
        rows = self._set_headers(raw)
        first_chunk = list(islice(rows, self.chunk_threshold))
        self._sampling_cache = first_chunk
        self._went_lazy = len(first_chunk) >= self.chunk_threshold
        return chain(first_chunk, rows) if self._went_lazy else iter(first_chunk)

    def _raw_rows(self):
        """Stream the raw CSV content as rows (text mode, encoding-safe)."""
        wrapper = open(self.data, 'r', encoding='utf-8', newline='')
        return csv.reader(wrapper)

    def _set_headers(self, rows):
        first = next(rows, None)
        if not first:
            self._headers = []
            return iter([])

        if self.has_headers:
            self._headers = first
            return rows
        else:
            self._headers = list(range(len(first)))
            return chain([first], rows)

    def _validate_column(self, col):
        if not self._headers:
            _ = self.get_rows()
        if col not in self._headers:
            raise ValueError(f"Column '{col}' not found. Available: {self._headers}")

    def apply_op(self, op):
        """Returns a function that applies the op in a SafeBinder context."""
        if op not in CMP_OPS:
            raise ValueError(f"Unsupported operator: {op}")
        return lambda *args: SafeBinder(args).bind(lambda xs: CMP_OPS[op](*xs))

    def make_row_accessor(self, idxA, idxB=None, coerce=False):
        """Create function to access one or two column values from a row."""
        def coerce_fn(v):
            try:
                return float(v) if '.' in v else int(v)
            except:
                return v

        def accessor(row):
            a = row[idxA]
            b = row[idxB] if idxB is not None else None
            return (coerce_fn(a), coerce_fn(b)) if coerce else (a, b)

        return accessor

    def make_predicate(self, accessor, apply_op, right_value=None):
        def predicate(row):
            a, b = accessor(row)
            rhs = right_value if right_value is not None else b
            return apply_op(a, rhs)()
        return predicate

    def gen_search_pred(self, colA, op, colB=None, value=None):
        self._validate_column(colA)
        if colB:
            self._validate_column(colB)

        idxA = self._headers.index(colA)
        idxB = self._headers.index(colB) if colB else None
        accessor = self.make_row_accessor(idxA, idxB, coerce=True)
        op_fn = self.apply_op(op)

        return self.make_predicate(accessor, op_fn, value)

    def run_search(self, predicate, rows=None):
        row_stream = rows if rows is not None else self.get_rows()
        return filter(predicate, row_stream)

    def combine_search(self, preds, logic='and'):
        logic = logic.lower()
        if logic not in LOGICAL_OPS:
            raise ValueError(f"Unsupported logic: {logic}")
        return reduce(LOGICAL_OPS[logic], preds)

    def apply_arith(self, colA, op, colB=None, value=None):
        """Apply arithmetic or concat op between two columns or column/value."""
        self._validate_column(colA)
        if colB:
            self._validate_column(colB)

        idxA = self._headers.index(colA)
        idxB = self._headers.index(colB) if colB else None
        op_fn = ARITH_OPS[op]

        def transform(row):
            a = row[idxA]
            b = row[idxB] if colB else value
            try:
                return op_fn(a, b)
            except Exception as e:
                return None

        return map(transform, self.get_rows())

    def mutate_and_cache(self, *args, output_col=None, **kwargs):
        """Apply mutation and store new column to sampling cache."""
        new_vals = list(self.apply_arith(*args, **kwargs))
        if output_col and output_col not in self._headers:
            self._headers.append(output_col)
        self._sampling_cache = [row + [val] for row, val in zip(self.get_rows(), new_vals)]
        return self._sampling_cache

    def track_memory(self, chunk):
        """Rough estimate of memory usage for a chunk or result set."""
        return sum(sys.getsizeof(row) for row in chunk)

    def is_lazy_mode(self):
        return self._went_lazy
