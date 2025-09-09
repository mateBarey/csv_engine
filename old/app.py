import argparse
from src.csvengine import ParseCSV  # or whatever your class is called
import shlex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--q', required=True)
    parser.add_argument('--fields', required=True)
    parser.add_argument('--diacritics-off', action='store_true')
    return parser.parse_args()

'''
criteria:

searching --q cafe matches row 2 whether you type “cafe” or “café”.

--q 'tag:csv "smart quotes"' returns rows 3 and 4 (because of “JSON extract” mention? your tokenizer choice may vary—explain).

row 5 ends up in snippets_errors.csv with a clear reason.

multiline row 6 parses as a single record.
'''

def custom_q_to_action(q: str, parser: ParseCSV):
    tokens = shlex.split(q)

    cmp_ops = parser._cmp_ops
    logic_ops = parser._logical_ops

    # Helpers
    is_logic_op     = lambda tok: tok.lower() in logic_ops
    is_negated      = lambda tok: tok.startswith("-")
    strip_prefix    = lambda tok: tok.lstrip("-")
    split_field_val = lambda tok: strip_prefix(tok).split(":", 1)

    # Default comparator
    default_cmp_op = "in"

    # Grab logical operator if present, default to "and"
    logic_fn = next(
        (logic_ops[tok.lower()] for tok in tokens if is_logic_op(tok)),
        logic_ops["and"]
    )

    # Parse field:value tokens
    parsed = [
        split_field_val(tok) + [is_negated(tok)]
        for tok in tokens if not is_logic_op(tok)
    ]

    # Convert each triple into a predicate
    predicates = map(
        lambda fld_val_neg: (
            logic_ops["not"](parser.gen_search_pred(fld_val_neg[0], default_cmp_op, value=fld_val_neg[1]))
            if fld_val_neg[2] else
            parser.gen_search_pred(fld_val_neg[0], default_cmp_op, value=fld_val_neg[1])
        ),
        parsed
    )

    # Combine all predicates using chosen logical operator (e.g., and, or)
    return reduce(logic_fn, predicates)


def main():
    args = parse_args()

    # ✅ You instantiate the engine using args.file
    parser = ParseCSV(args.file, has_headers=True)

    # ✅ Get rows
    rows = parser.get_rows()

    # ✅ Normalize if --diacritics-off is present
    if args.diacritics_off:
        rows = normalize_rows(rows, fields=args.fields.split(','))

    # ✅ Parse query DSL to a predicate function
    pred = query_to_predicate(args.q, args.fields.split(','))

    # ✅ Run filtered search
    result = filter(pred, rows)

    # ✅ Output: stdout + jsonl
    emit_results(result)

if __name__ == '__main__':
    main()
