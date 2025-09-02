# Test_Repo
loadJson to Postges 

python -m venv ./.venv

.venv/bin/activate

pip install -r requirements.txt

python3 loadJson.py  --path "/Users/aps/Documents/GitHub/colorDataToJson/Data" --json_file "printer.cie.json" --db "postgresql+psycopg2://pgadmin:postgres@10.211.55.9:5432/postgres" --dry-run --no-db-check --schema "color_measurement" --debug --verbose

 python3 loadJson.py  --path "/Users/aps/Documents/GitHub/jsonToPostgres/Data" --json_file "printer.cie.json" --db "postgresql+psycopg2://pgadmin:postgres@10.211.55.9:5432/postgres"  --schema "color_measurement"  --verbose     


## Logs

Relative paths passed to `--log-file` are resolved against the script's source folder. On Windows, a path starting with `\` is treated as absolute (e.g., `\temp\my.log`).


  File "/Users/aps/Documents/GitHub/jsonToPostgres/loadJson.py", line 447, in <module>
    main()
    ~~~~^^
  File "/Users/aps/Documents/GitHub/jsonToPostgres/loadJson.py", line 414, in main
    raise FileNotFoundError(f"JSON file not found: {json_file}")
FileNotFoundError: JSON file not found: /Users/aps/Documents/GitHubjsonToPostgres/Data/printer.cie.json
(.venv) aps@macbookpro jsonToPostgres % 

## Changelog
**2025-09-01 — Version 1.0.1**
- Logging now writes to `(source)/Logs` with daily filename `loadJson_YYYY-MM-DD.log`.
- New flags: `--log-level` and `--log-file`.
- Relative `--log-file` resolves against the script folder; on Windows, a leading `\` or `\\` is treated as absolute.
- Redacts passwords in DB URLs in logs.