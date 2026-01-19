

import duckdb
import os
import json
import re

def derive_table_name(file_path):
    """Derives a table name from a file path."""
    base_name = os.path.basename(file_path)
    table_name, _ = os.path.splitext(base_name)
    table_name = table_name.lower()
    table_name = re.sub(r'[^a-z0-9]+', '_', table_name)
    if table_name.startswith(tuple('0123456789')):
        table_name = f't_{table_name}'
    return table_name

def main():
    """Main function to load data into DuckDB."""
    db_path = './.db/schema_mathcing.db'
    db_name = 'schema_mathcing'
    schema_name = 'main'
    script_dir = './.script/01-SQL-DDL'
    
    os.makedirs(script_dir, exist_ok=True);

    data_sources = [
        'data/samples/green_tripdata_2025-01_sample_10000.parquet',
        'data/samples/yellow_tripdata_2025-01_sample_10000.parquet',
        'data/samples/hvfhs_tripdata_2025-01_sample_10000.parquet'
    ]

    con = duckdb.connect(database=db_path, read_only=False)
    
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
    con.execute(f"USE {schema_name};")

    plan = []
    ddl_script = f"CREATE SCHEMA IF NOT EXISTS {schema_name};\nUSE {schema_name};\n\n"
    verify_script = ""
    run_all_py_script = f"""
import duckdb

def run_all():
    con = duckdb.connect(database='{db_path}', read_only=False)
"""

    for source in data_sources:
        table_name = derive_table_name(source)
        plan.append({
            "source": source,
            "table_name": table_name,
            "type": "parquet",
            "mode": "execute"
        })

        # DDL and Load Script
        load_script = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{source}');"
        ddl_script += f"DROP TABLE IF EXISTS {table_name};\n{load_script}\n\n"
        
        with open(os.path.join(script_dir, f'02_load_{table_name}.sql'), 'w') as f:
            f.write(load_script)

        # Verification Script
        verify_script += f"SELECT COUNT(*) FROM {table_name};\n"
        verify_script += f"SELECT * FROM {table_name} LIMIT 5;\n\n"
        
        # Python Script
        run_all_py_script += f"""
    con.execute("""
    {load_script}
    """)
"""

    with open(os.path.join(script_dir, '00_PLAN.md'), 'w') as f:
        f.write(json.dumps(plan, indent=4))

    with open(os.path.join(script_dir, '01_sql_ddl.sql'), 'w') as f:
        f.write(ddl_script)

    with open(os.path.join(script_dir, '03_verify.sql'), 'w') as f:
        f.write(verify_script)
        
    with open(os.path.join(script_dir, 'run_all.py'), 'w') as f:
        f.write(run_all_py_script)

    # Documentation
    doc = f"""
        # SQL DDL Documentation

        ## Goal
        This document describes the SQL DDL scripts generated to load data into the '{db_name}' database.

        ## Warnings
        - The database name seems to have a typo: 'schema_mathcing' instead of 'schema_matching'.

        ## Tables
    """
    for item in plan:
        doc += f"- `{item['table_name']}`: Loaded from `{item['source']}`\n"
    
    with open(os.path.join(script_dir, '01_SQL_DDL.MD'), 'w') as f:
        f.write(doc)

    # Run all script
    run_all_sh = f"""
#!/bin/bash
duckdb ./.db/schema_mathcing.db < ./.script/01-SQL-DDL/01_sql_ddl.sql
duckdb ./.db/schema_mathcing.db < ./.script/01-SQL-DDL/03_verify.sql
    """
    with open(os.path.join(script_dir, 'run_all.sh'), 'w') as f:
        f.write(run_all_sh)

    # Execute the DDL
    con.execute(ddl_script)

    # Final JSON Report
    report = {
        "status": "success",
        "database": db_name,
        "schema": schema_name,
        "tables_loaded": [item['table_name'] for item in plan]
    }
    print(json.dumps(report, indent=4))

    con.close()

if __name__ == "__main__":
    main()

