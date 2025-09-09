import csv
import pandas as pd
import polars as pl
import os
from pathlib import Path
import re
from itertools import islice, chain
from functools import reduce , wraps
import io
from typing import Dict, List, Optional
from operator import eq, ne, lt, le, gt, ge, add, sub, mul, truediv
from operator import concat as str_concat
import sys
import unicodedata
import argparse

os.chdir('/home/matebarey/Downloads/repos/mozity')

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Search CSV with field filtering and optional diacritics removal.")

    parser.add_argument('--file', required=True, help='Path to the CSV file')
    parser.add_argument('--q', required=True, help='Search query string (e.g., "cafe")')
    parser.add_argument('--fields', required=True, help='Comma-separated list of fields to search (e.g., title,body)')
    parser.add_argument('--diacritics-off', action='store_true', help='Remove diacritics when matching text')

    return parser.parse_args()

file_name  = "football.csv"
file_name2 = "weather.csv"
#phase 1  load and understand

# pandas polars default is interpret utf8 if they cant decoe  they thorw code error
def clean_csv_bytes(b: bytes) -> bytes :
    fixes = {
        'crlf': (rb'\r\n', b'\n'),
        'cr': (rb'\r(?!\n)', b'\n'),
        'bom_start': (rb'^\xef\xbb\xbf', b''),
        'bom_interior': (b'\xef\xbb\xbf', b''),
        'null': (b'\x00', b''),
        'sep_header': (rb'(?i)^sep=.*?\n', b''),
        'curly_quotes_open': (b'\xe2\x80\x9c', b'"'),
        'curly_quotes_close': (b'\xe2\x80\x9d', b'"'),
        'curly_single_open': (b'\xe2\x80\x98', b"'"),
        'curly_single_close': (b'\xe2\x80\x99', b"'"),
        'grave_accent': (b'\x60', b"'"),
        'acute_accent': (b'\xc2\xb4', b"'"),
        'unescaped_quotes': (rb'(?<!^)(?<!,)"(?!,)(?!$)(?!")', b'""')
    }

    # Single comprehension: count and apply fixes if count > 0
    return reduce(
        lambda content, fix: re.sub(fix[0], fix[1], content),
        [fix for fix in fixes.values() if len(re.findall(fix[0], b)) > 0],
        b
    )
# detect and stirp bom  with utf-8-sig
def mutating(fn):
    @wraps(fn)
    def wrapper(self, *args, mutate=False, output_col=None, **kwargs):
        base_rows = self.get_rows()
        result_gen = list(fn(self, *args, **kwargs))  # Force evaluation

        if mutate:
            if not output_col:
                raise ValueError("Must specify output_col when mutating")
            if self.has_headers and output_col not in self._headers:
                self._headers.append(output_col)

            return (row + [val] for row, val in zip(base_rows, result_gen))
        else:
            return result_gen
    return wrapper

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def logical_and(pred1, pred2):
    def combined(row):
        return pred1(row) and pred2(row)
    return combined

def logical_in(pred1, pred2):
    def combined(row):
        return pred1(row) in pred2(row)
    return combined


def str_concat(a,b):
    return str(a) + str(b)
def logical_or(pred1, pred2):
    def combined(row):
        return pred1(row) or pred2(row)
    return combined

def logical_not(pred):
    def inverted(row):
        return not pred(row)
    return inverted

class SafeBinder:
    def __init__(self, value=None, status=None):
        self.value = value
        self.status = status

    def bind(self,f):
        if self.status is not None :
            return SafeBinder(None,self.status)

        try:
            result = f(self.value)
            return SafeBinder(result,None)
        except Exception as e:
            return SafeBinder(None, e)

    def __call__(self):
        return self.value

class ParseCSV:
    def __init__(self,file_path,has_headers=True, chunk_threshold=1000):
        self.data = file_path
        self.has_headers = has_headers
        self.chunk_threshold = chunk_threshold
        self._sampling_cache = []
        self._went_lazy = False
        self._cmp_ops   = {"eq": eq, "==": eq, "ne": ne, "!=": ne,
                           "lt": lt, "<": lt, "le": le, "<=": le,
                            "in": lambda a,b: SafeBinder(b in a),
                            "not in": lambda a,b: SafeBinder(b not in a),
                           "gt": gt, ">": gt, "ge": ge, ">=": ge}
        self._arith_ops = {"+": add, "-": sub, "*": mul, "/": truediv, "concat": str_concat}
        self._logical_ops = {
            "and": logical_and,
            "or": logical_or,
            "not": logical_not,

        }


        self.encodings = ['utf-8-sig', 'utf-8', 'latin1']
        self._headers = []

    def apply_op(self, op):
        if op in self._cmp_ops:
            return lambda *args: SafeBinder(args).bind(lambda xs: self._cmp_ops[op](*xs))
        raise ValueError(f"Unsupported operator: {op}")

    def _identity(self, x):
        return x

    def gen_search(self, colA, op, colB=None, value=None):
        pred = self.gen_search_pred(colA, op, colB, value)
        return self.run_search(pred)

    def run_search(self, predicate, rows=None):
        row_stream = rows if rows is not None else self.get_rows()
        return filter(predicate, row_stream)

    def make_predicate(self, accessor, apply_op, right_value=None):
        def predicate(row):
            a, b = accessor(row)
            rhs = right_value if right_value is not None else b
            return apply_op(a, rhs)()
        return predicate

    def make_coercer(self):
        def _coerce(val):
            return float(val) if '.' in val else int(val)

        return self.apply_op(_coerce)

    def cli_q_to_action(q: str, diacritics_off=False) :

        pass

    def gen_search_pred(self, colA, op, colB=None, value=None):
        self._validate_search(colA, op, colB, value)
        self._validate_column(colA)
        if colB:
            self._validate_column(colB)

        idxA = self._headers.index(colA)
        idxB = self._headers.index(colB) if colB else None

        accessor = self.make_row_accessor(idxA, idxB, coerce=True)
        op_applier = self.apply_op(self._cmp_ops[op])

        return self.make_predicate(accessor, op_applier, value)


    def make_row_accessor(self, idxA, idxB=None, coerce=False):
        coerce_fn = self.make_coercer() if coerce else None

        def accessor(row):
            if coerce_fn is None:
                a = row[idxA]
                b = row[idxB] if idxB is not None else None
            else:
                a = coerce_fn(row[idxA])()
                b = coerce_fn(row[idxB])() if idxB is not None else None
            return a, b
        return accessor

    def is_lazy_mode(self):
        return self._went_lazy

    def normalize_text(text: str, *, lowercase=True, remove_accents=True) -> str:
        if remove_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
        if lowercase:
            text = text.lower()
        return text

    def _validate_search(self, colA, op, colB, value):
        if not colA or not op:
            raise ValueError("colA and op are required")
        if colB is None and value is None:
            raise ValueError("Either colB or value must be provided")
        if colB is not None and value is not None:
            raise ValueError("Provide either colB or value, not both")


            # there is no value  but there is col A and col B a
    @mutating
    def apply_arith(self, colA, op, colB=None, value=None):
        if not self._headers:
            _ = self.get_rows()

        idxA = self._headers.index(colA)
        idxB = self._headers.index(colB) if colB else None
        op_fn = self._arith_ops[op]  # <- JUST the raw function

        is_math = op in {"+", "-", "*", "/"}
        coerce = self.make_coercer() if is_math else self._identity

        def transformer(row):
            a_raw = row[idxA]
            b_raw = row[idxB] if colB else value

            try:
                a = coerce(a_raw)() if is_math else a_raw
                b = coerce(b_raw)() if (colB and is_math) else b_raw
            except Exception as e:
                print(f"[WARN] Coercion failed: {e} | Row: {row}")
                return None

            try:
                return op_fn(a, b)
            except Exception as e:
                print(f"[WARN] Operation failed: {a} {op} {b} | Error: {e}")
                return None

        return (transformer(row) for row in self.get_rows())

    # NEW: decide delimiter from one line by max char count among common delims
    def _sniff_delimiter(self, sample: str) -> str:
        def try_sniff(s):
            return csv.Sniffer().sniff(s).delimiter
        return SafeBinder(sample).bind(try_sniff).bind(str).__call__() or ','

    def combine_search(self, preds, logic='and'):
        logic = logic.lower()
        op_func = self._logical_ops[logic]
        return op_func(*preds)


    def _validate_column(self, col_name: str):
        if not self._headers:
            _ = self.get_rows()  # ensures headers are loaded
        if col_name not in self._headers:       # <-- fix: remove self.self
            raise ValueError(f"Column '{col_name}' not found. Available: {self._headers}")

    def get_column(self, col_name):
        if not self._headers:
            _ = self.get_rows()  # ensure headers are loaded
        if col_name not in self._headers:
            raise ValueError(f"Column '{col_name}' not found. Available: {self._headers}")

        idx = self._headers.index(col_name)
        return [row[idx] for row in self.get_rows()]

    def mutate_and_cache(self, *args, **kwargs):
        result = list(self.apply_arith(*args, **kwargs))
        self._sampling_cache = result
        return result

    def _validate_operator(self, op):
        """Validate that operator is callable and supported"""
        if not callable(op):
            raise ValueError(f"Operator must be callable, got {type(op)}")

        # Optional: Check if it's a known comparison operator

        if op not in self._cmp_ops:
            # Warn but don't fail - user might have custom operators
            print(f"Warning: Uncommon operator {op}. Use standard comparison operators for best results.")

    def get_rows(self):  # lazy after reading file into memory
        raw = self._raw_rows()
        rows = self._set_headers(raw)  # ensure headers are set here
        first_chunk = list(islice(rows, self.chunk_threshold))
        self._sampling_cache = first_chunk
        self._went_lazy = len(first_chunk) >= self.chunk_threshold
        return chain(iter(first_chunk), rows) if self._went_lazy else iter(first_chunk)

    def _set_headers(self, rows):
        first = next(rows, None)
        if not first:
            self._headers = []
            return iter([])

        # User specified or default to numeric
        if hasattr(self, 'has_headers') and self.has_headers:
            self._headers = first
            return rows
        else:
            self._headers = list(range(len(first)))
            return chain([first], rows)


    def cleaned_buffer(self):
        def try_load(fpath):
            with open(fpath, 'rb') as f:
                return f.read()

        safe_raw = SafeBinder(self.data).bind(try_load).bind(clean_csv_bytes)

        text = safe_raw.bind(lambda b: b.decode('utf-8', errors='replace')).__call__() or ""

        if not getattr(self, 'delimiter', None):
            self.delimiter = self._sniff_delimiter(text[:1024])

        return io.StringIO(text)


    def _raw_rows(self):
        def try_encoding(enc):
            return SafeBinder(enc).bind(lambda e: io.TextIOWrapper(open(self.data, 'rb'), encoding=e, newline=''))

        wrapper_binder = next((b for b in (try_encoding(e) for e in self.encodings) if b.status is None), None)

        if wrapper_binder is None:
            wrapper = io.TextIOWrapper(open(self.data, 'rb'), encoding='latin1', errors='replace', newline='')
        else:
            wrapper = wrapper_binder()

        # Let csv.reader do the heavy lifting — no line-by-line pre-clean
        return csv.reader(wrapper)

if __name__ == '__main__':
    parser_csv = ParseCSV(file_path=file_name,has_headers=True)
    new_col = list(parser_csv.mutate_and_cache("Goals","-","Goals Allowed",mutate=True,output_col="sum"))
    #new_col = list(parser_csv("Goals","-","Goals Allowed"))
    double_points = list(parser_csv.apply_arith("Points", "*", value=2))
    concat_col = list(parser_csv.apply_arith("Team", "concat", value=" FC"))
    p1 = parser_csv.gen_search_pred("Goals", ">", value=50)
    p2 = parser_csv.gen_search_pred("Goals Allowed", "<", value=40)
    combined = parser_csv.combine_search([p1, p2], logic="and")
    view = parser_csv.run_search(combined)
    print(list(view))
