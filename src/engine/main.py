if __name__ == '__main__':
    args = parse_args()
    engine = ParseCSV(args.file, has_headers=True)
    pred = custom_q_to_action(args.q, engine)
    results = engine.run_search(pred)
    emit_results(results)
