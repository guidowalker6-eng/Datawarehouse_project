from urllib.parse import quote_plus
from mage_ai.data_preparation.shared.secrets import get_secret_value
import sqlalchemy as sa
import pandas as pd
from datetime import datetime, date
from io import StringIO
import csv
import smbclient
import time
import traceback
import re

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


CONTROL_TARGET_COLUMN = 'target_table_name'
SAFE_IDENTIFIER_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def get_target_table_name(item):
    target_table_name = item.get('target_table_name')
    if target_table_name and str(target_table_name).strip():
        return str(target_table_name).strip()

    target_name = item.get('target_name')
    if target_name and str(target_name).strip():
        return str(target_name).strip()

    db_name = item.get('db_name')
    table_name = item.get('table_name')
    if item.get('type') == 'file':
        return table_name
    if db_name and table_name:
        return f"{db_name}_{table_name}"
    return table_name


def psql_copy(table, conn, keys, data_iter):
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        buf = StringIO()
        csv.writer(buf).writerows(data_iter)
        buf.seek(0)
        cols = ', '.join([f'"{k}"' for k in keys])
        tbl = f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
        cur.copy_expert(f'COPY {tbl} ({cols}) FROM STDIN WITH CSV', buf)


def smb_to_unc(windows_path, smb_host, smb_share):
    path = windows_path.replace('\\', '/')
    if ':/' in path:
        path = path.split(':/', 1)[1]
    return f'//{smb_host}/{smb_share}/{path}'


def _table_exists_in_bronze(dst_engine, table_name):
    with dst_engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'bronze'
              AND table_name = :tbl
            LIMIT 1
        """), {'tbl': table_name}).fetchone()
    return row is not None


def _assert_safe_identifier(identifier, label):
    value = str(identifier or '').strip()
    if not value:
        raise ValueError(f"Identificador vacio para {label}")
    if not SAFE_IDENTIFIER_PATTERN.match(value):
        raise ValueError(f"Identificador invalido para {label}: {value}")
    return value


def _quote_ident(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def _get_target_bronze_columns(dst_engine, target_table):
    safe_target_table = _assert_safe_identifier(target_table, 'target_table')
    with dst_engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'bronze'
              AND table_name = :tbl
        """), {'tbl': safe_target_table}).fetchall()

    out = set()
    for row in rows:
        row_map = getattr(row, '_mapping', None)
        col_name = row_map.get('column_name') if row_map else row[0]
        if col_name:
            out.add(str(col_name))
    return out


def _sync_missing_columns_in_bronze(dst_engine, target_table, source_columns_with_types):
    safe_target_table = _assert_safe_identifier(target_table, 'target_table')
    existing_cols = _get_target_bronze_columns(dst_engine, safe_target_table)
    missing_cols = [col for col in source_columns_with_types.keys() if col not in existing_cols]

    if not missing_cols:
        return []

    added = []
    quoted_table = _quote_ident(safe_target_table)

    with dst_engine.begin() as ddl_conn:
        for col in missing_cols:
            safe_col = _assert_safe_identifier(col, f'columna destino {safe_target_table}')
            sql_type = source_columns_with_types.get(safe_col) or 'TEXT'
            quoted_col = _quote_ident(safe_col)
            ddl = sa.text(
                f"ALTER TABLE bronze.{quoted_table} "
                f"ADD COLUMN IF NOT EXISTS {quoted_col} {sql_type}"
            )
            run_with_retry(
                lambda ddl=ddl: ddl_conn.execute(ddl),
                f"add_missing_column_{safe_target_table}.{safe_col}"
            )
            added.append({'column': safe_col, 'sql_type': sql_type})

    if added:
        details = ', '.join([f"{x['column']}:{x['sql_type']}" for x in added])
        print(f"  [SCHEMA] Columnas agregadas en bronze.{safe_target_table}: {details}")

    return added


def _pandas_dtype_to_sql(series):
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    if pd.api.types.is_integer_dtype(dtype):
        return 'BIGINT'
    if pd.api.types.is_float_dtype(dtype):
        return 'DOUBLE PRECISION'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    if pd.api.types.is_timedelta64_dtype(dtype):
        return 'INTERVAL'
    return 'TEXT'


def _build_columns_with_types_from_chunk(chunk_df):
    result = {}
    for column_name in chunk_df.columns:
        safe_column_name = _assert_safe_identifier(column_name, 'chunk_column')
        result[safe_column_name] = _pandas_dtype_to_sql(chunk_df[column_name])
    return result


def _coerce_integer_columns(df):
    """
    Pandas lee columnas enteras con NULLs como float64 (ej: 1.0, 2.0).
    PostgreSQL rechaza esos valores en columnas BIGINT/INTEGER via COPY.
    Esta funcion convierte float64 que son enteros a Int64 nullable.
    """
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null == non_null.astype('int64')).all():
                df[col] = df[col].astype(pd.Int64Dtype())
    return df


def _prepare_full_refresh_target(dst_engine, src_engine, source_table, target_table):
    target_exists = run_with_retry(
        lambda: _table_exists_in_bronze(dst_engine, target_table),
        f"check_target_exists_{source_table}->{target_table}"
    )

    if target_exists:
        with dst_engine.begin() as ddl_conn:
            run_with_retry(
                lambda: ddl_conn.execute(sa.text(
                    f'TRUNCATE TABLE bronze."{target_table}"'
                )),
                f"truncate_target_{source_table}->{target_table}"
            )
    else:
        # Crea estructura aunque la fuente venga vacia (0 filas).
        empty_df = run_with_retry(
            lambda: pd.read_sql(sa.text(f'SELECT * FROM "{source_table}" LIMIT 0'), src_engine),
            f"leer_estructura_source_{source_table}"
        )
        run_with_retry(
            lambda: empty_df.to_sql(
                name=target_table,
                con=dst_engine,
                schema='bronze',
                if_exists='fail',
                index=False
            ),
            f"crear_tabla_vacia_{source_table}->{target_table}"
        )

    with dst_engine.begin() as ddl_conn:
        run_with_retry(
            lambda: ddl_conn.execute(sa.text(
                f'ALTER TABLE bronze."{target_table}" '
                f'ADD COLUMN IF NOT EXISTS _row_id BIGSERIAL'
            )),
            f"agregar_row_id_{source_table}->{target_table}"
        )


RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 5


def is_retryable_error(exc):
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    if isinstance(exc, (
        sa.exc.OperationalError,
        sa.exc.TimeoutError,
        sa.exc.DisconnectionError,
        sa.exc.InterfaceError,
    )):
        return True

    if isinstance(exc, sa.exc.DBAPIError) and getattr(exc, 'connection_invalidated', False):
        return True

    err_text = str(getattr(exc, 'orig', exc)).lower()
    transient_tokens = (
        'timeout',
        'timed out',
        'could not connect',
        'connection refused',
        'connection reset',
        'server closed the connection',
        'temporarily unavailable',
        'too many connections',
        'deadlock detected',
    )
    return any(token in err_text for token in transient_tokens)


def run_with_retry(operation, operation_name, max_attempts=RETRY_MAX_ATTEMPTS, base_wait_seconds=RETRY_BASE_SECONDS):
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            retryable = is_retryable_error(exc)
            is_last_attempt = attempt == max_attempts

            if (not retryable) or is_last_attempt:
                reason = 'no_retry_error_permanente' if not retryable else 'retries_agotados'
                print(f"  [RETRY][{reason}] {operation_name} | intento={attempt}/{max_attempts} | error={str(exc)[:180]}")
                raise

            wait_seconds = base_wait_seconds * (2 ** (attempt - 1))
            print(
                f"  [RETRY] {operation_name} | intento={attempt}/{max_attempts} "
                f"| espera={wait_seconds}s | error={str(exc)[:180]}"
            )
            time.sleep(wait_seconds)


def get_pipeline_extraction_id(manifest):
    for item in manifest:
        extraction_id = item.get('extraction_id')
        if extraction_id:
            return extraction_id
    return 'SIN_ID'


def build_pipeline_log_rows(manifest, audit_summary, extraction_id, fatal_error=None, fatal_trace=None):
    manifest_issues = []
    for item in manifest:
        item_status = str(item.get('status') or '').upper()
        if item_status and item_status != 'READY':
            manifest_issues.append({
                'db': item.get('db_name', '?'),
                'tabla': item.get('target_table_name') or item.get('table_name', '?'),
                'status': item.get('status'),
                'error': item.get('error', ''),
            })

    load_status = 'ERROR' if manifest_issues else 'SUCCESS'
    load_message = None
    if manifest_issues:
        details = [
            f"{issue['db']}.{issue['tabla']}={issue['status']} {issue['error']}".strip()
            for issue in manifest_issues[:10]
        ]
        load_message = ' | '.join(details)

    transformer_status = 'SUCCESS' if extraction_id != 'SIN_ID' else 'ERROR'
    transformer_message = None if transformer_status == 'SUCCESS' else 'No se encontro extraction_id en el manifest.'

    export_errors = []
    for row in audit_summary:
        row_status = str(row.get('status') or '').upper()
        if 'ERROR' in row_status:
            export_errors.append({
                'db': row.get('db', '?'),
                'tabla': row.get('target_table') or row.get('tabla', '?'),
                'status': row.get('status'),
                'error': row.get('error', ''),
            })

    exporter_status = 'ERROR' if fatal_error or export_errors else 'SUCCESS'
    exporter_message = None
    exporter_trace = None

    if fatal_error:
        exporter_message = str(fatal_error)
        exporter_trace = fatal_trace
    elif export_errors:
        details = [
            f"{issue['db']}.{issue['tabla']}={issue['status']} {issue['error']}".strip()
            for issue in export_errors[:10]
        ]
        exporter_message = ' | '.join(details)

    timestamp = datetime.now()
    return [
        {
            'extraction_id': extraction_id,
            'activity_pipeline_name': 'bronze_load',
            'estatus': load_status,
            'error_message': load_message,
            'error_trace': None,
            'timestamp': timestamp,
        },
        {
            'extraction_id': extraction_id,
            'activity_pipeline_name': 'bronze_transformer',
            'estatus': transformer_status,
            'error_message': transformer_message,
            'error_trace': None,
            'timestamp': timestamp,
        },
        {
            'extraction_id': extraction_id,
            'activity_pipeline_name': 'bronze_write_exporter',
            'estatus': exporter_status,
            'error_message': exporter_message,
            'error_trace': exporter_trace,
            'timestamp': timestamp,
        },
    ]


def insert_pipeline_logs(dst_engine, log_rows):
    with dst_engine.begin() as conn:
        for row in log_rows:
            run_with_retry(
                lambda row=row: conn.execute(sa.text("""
                    INSERT INTO bronze._error_message_pipeline
                        (extraction_id, activity_pipeline_name, estatus, error_message, error_trace, "timestamp")
                    VALUES
                        (:extraction_id, :activity_pipeline_name, :estatus, :error_message, :error_trace, :timestamp)
                """), row),
                f"insert_pipeline_log_{row['activity_pipeline_name']}"
            )


def insert_execution_times(dst_engine, timing_rows):
    if not timing_rows:
        return
    with dst_engine.begin() as conn:
        for row in timing_rows:
            try:
                conn.execute(sa.text("""
                    INSERT INTO bronze._pipeline_execution_times
                        (execution_id, pipeline_name, activity_name, started_at, finished_at, duration_ms, status, error_message)
                    VALUES
                        (:execution_id, :pipeline_name, :activity_name, :started_at, :finished_at, :duration_ms, :status, :error_message)
                """), row)
            except Exception as te:
                print(f"WARN: No se pudo registrar tiempo actividad {row.get('activity_name')}: {te}")


@data_exporter
def export_data(manifest, *args, **kwargs):
    if manifest is None:
        print("export_data: input None -> []")
        return []

    if isinstance(manifest, dict):
        manifest = [manifest]

    if not isinstance(manifest, list):
        print(f"export_data: tipo inesperado {type(manifest)} -> []")
        return []

    manifest = [x for x in manifest if isinstance(x, dict)]
    if len(manifest) == 0:
        print("export_data: manifest vacio")
        return []

    # Extraer timing metadata pasado desde LOAD y TRANSFORMER (no son items reales)
    _timing_from_manifest = [x for x in manifest if x.get('_timing_meta')]
    manifest = [x for x in manifest if not x.get('_timing_meta')]
    _write_started_at = datetime.now()

    dst_engine = None
    audit_summary = []
    pipeline_fatal_error = None
    pipeline_fatal_trace = None
    pipeline_extraction_id = get_pipeline_extraction_id(manifest)
    run_ts = datetime.now().strftime('%Y%m%d-%H%M')
    hoy = date.today()
    total = len(manifest)
    CHUNK_SIZE = 10_000

    try:
        src_host = get_secret_value('BDWORKFLOW_HOST')
        src_user = get_secret_value('BDWORKFLOW_USER')
        src_pass = get_secret_value('BDWORKFLOW_PASSWORD')

        dst_host = get_secret_value('BDMANAGER_HOST')
        dst_user = get_secret_value('BDMANAGER_USER')
        dst_pass = get_secret_value('BDMANAGER_PASSWORD')

        smb_host = get_secret_value('smb-server')
        smb_share = get_secret_value('smb-share')
        smb_user = get_secret_value('smb-user')
        smb_pass = get_secret_value('smb-password')

        smbclient.ClientConfig(username=smb_user, password=smb_pass)

        dst_engine = sa.create_engine(
            f"postgresql://{quote_plus(dst_user)}:{quote_plus(dst_pass)}@{dst_host}:5432/postgres",
            connect_args={'client_encoding': 'utf8'}
        )

        for idx, item in enumerate(manifest, start=1):
            base_extraction_id = item.get('extraction_id', 'SIN_ID')
            extraction_id_suffix = (
                base_extraction_id.split('-')[-1]
                if base_extraction_id and '-' in base_extraction_id
                else 'RUN'
            )

            _db = item.get('db_name', '?')
            _tbl = item.get('table_name', '?')
            _target_tbl = get_target_table_name(item)
            print(f"[{idx}/{total}] {_db}.{_tbl} -> {_target_tbl} ...")

            if item.get('status') != 'READY':
                audit_summary.append({
                    'db': _db,
                    'tabla': _tbl,
                    'target_table': _target_tbl,
                    'status': item.get('status'),
                    'error': item.get('error', '')
                })
                continue

            if not _target_tbl:
                audit_summary.append({
                    'db': _db,
                    'tabla': _tbl,
                    'target_table': None,
                    'status': 'ERROR_EXPORT',
                    'error': 'Item sin target_table_name valido'
                })
                continue

            with dst_engine.connect() as ctrl:
                ya_cargado = run_with_retry(
                    lambda: ctrl.execute(sa.text(f"""
                        SELECT {CONTROL_TARGET_COLUMN}, load_mode, watermark_col, watermark_value
                        FROM bronze._control_carga
                        WHERE db_name = :db
                          AND table_name = :tbl
                          AND fecha_carga = :hoy
                          AND status = 'EXPORTADO'
                        ORDER BY fin DESC NULLS LAST
                        LIMIT 1
                    """), {
                        'db': item.get('db_name'),
                        'tbl': item.get('table_name'),
                        'hoy': hoy
                    }).fetchone(),
                    f"check_idempotencia_{item.get('db_name')}.{item.get('table_name')}->{_target_tbl}"
                )

            if ya_cargado:
                target_ya_cargado = None
                load_mode_prev = None
                watermark_col_prev = None
                watermark_value_prev = None
                try:
                    row_map = getattr(ya_cargado, '_mapping', None)
                    if row_map:
                        target_ya_cargado = row_map.get(CONTROL_TARGET_COLUMN)
                        load_mode_prev = row_map.get('load_mode')
                        watermark_col_prev = row_map.get('watermark_col')
                        watermark_value_prev = row_map.get('watermark_value')
                    else:
                        target_ya_cargado = ya_cargado[0] if len(ya_cargado) > 0 else None
                        load_mode_prev = ya_cargado[1] if len(ya_cargado) > 1 else None
                        watermark_col_prev = ya_cargado[2] if len(ya_cargado) > 2 else None
                        watermark_value_prev = ya_cargado[3] if len(ya_cargado) > 3 else None
                except Exception:
                    target_ya_cargado = None
                    load_mode_prev = None
                    watermark_col_prev = None
                    watermark_value_prev = None
                detalle = 'ya existe registro EXPORTADO para db/table/fecha_carga'
                if target_ya_cargado and str(target_ya_cargado) != str(_target_tbl):
                    detalle = (
                        f"ya existe registro EXPORTADO para db/table/fecha_carga "
                        f"con target_table_name={target_ya_cargado}"
                    )
                audit_summary.append({
                    'db': item.get('db_name'),
                    'tabla': item.get('table_name'),
                    'target_table': _target_tbl,
                    'status': 'SALTADO_IDEMPOTENCIA',
                    'filas': 0,
                    'error': detalle,
                    'load_mode': load_mode_prev,
                    'watermark_col': watermark_col_prev,
                    'watermark_value': watermark_value_prev,
                    'watermark_val_inicio': watermark_value_prev,
                })
                continue

            if item.get('type') == 'database':
                db_name = item.get('db_name')
                table = item.get('table_name')
                target_table = _target_tbl
                start = datetime.now()
                total_rows = 0

                if not db_name or not table or not target_table:
                    audit_summary.append({
                        'db': db_name,
                        'tabla': table,
                        'target_table': target_table,
                        'status': 'ERROR_EXPORT',
                        'error': 'Item sin db_name, table_name o target_table_name'
                    })
                    continue

                source_entity = item.get('source_entity', db_name.upper())
                source_prefix = item.get('source_prefix', str(db_name)[:3].upper())
                item_extraction_id = f"{source_prefix}-{run_ts}-{extraction_id_suffix}"

                try:
                    _assert_safe_identifier(table, f'tabla fuente {db_name}')
                    _assert_safe_identifier(target_table, f'tabla destino {db_name}.{table}')

                    src_engine = sa.create_engine(
                        f"postgresql://{quote_plus(src_user)}:{quote_plus(src_pass)}@{src_host}:5432/{db_name}",
                        connect_args={'connect_timeout': 5, 'client_encoding': 'utf8'}
                    )

                    watermark_col = item.get('watermark_col', '') or ''
                    watermark_val = None

                    with dst_engine.connect() as ctrl:
                        last_ok = run_with_retry(
                            lambda: ctrl.execute(sa.text(f"""
                                SELECT watermark_col, watermark_value
                                FROM bronze._control_carga
                                WHERE db_name = :db
                                  AND table_name = :tbl
                                  AND {CONTROL_TARGET_COLUMN} = :target_tbl
                                  AND status = 'EXPORTADO'
                                ORDER BY fin DESC NULLS LAST
                                LIMIT 1
                            """), {
                                'db': db_name,
                                'tbl': table,
                                'target_tbl': target_table,
                            }).fetchone(),
                            f"leer_ultimo_control_{db_name}.{table}->{target_table}"
                        )

                    if last_ok and last_ok[0] and last_ok[1]:
                        watermark_col = last_ok[0]
                        watermark_val = last_ok[1]

                        with src_engine.connect() as chk:
                            col_ok = run_with_retry(
                                lambda: chk.execute(sa.text("""
                                    SELECT 1
                                    FROM information_schema.columns
                                    WHERE table_name = :tbl
                                      AND column_name = :col
                                """), {
                                    'tbl': table,
                                    'col': watermark_col
                                }).fetchone(),
                                f"validar_watermark_col_{db_name}.{table}"
                            )

                        if col_ok:
                            load_mode = 'incremental'
                            query = sa.text(
                                f'SELECT * FROM "{table}" WHERE "{watermark_col}" > :wv'
                            ).bindparams(wv=watermark_val)
                            if_exists_mode = 'append'
                        else:
                            print(f"  [WARN] '{watermark_col}' no existe en fuente -> full_refresh")
                            load_mode = 'full_refresh'
                            watermark_col = ''
                            watermark_val = None
                            query = sa.text(f'SELECT * FROM "{table}"')
                            if_exists_mode = 'append'
                    else:
                        load_mode = 'full_refresh'
                        query = sa.text(f'SELECT * FROM "{table}"')
                        if_exists_mode = 'append'

                        if watermark_col:
                            with src_engine.connect() as chk:
                                col_ok = run_with_retry(
                                    lambda: chk.execute(sa.text("""
                                        SELECT 1
                                        FROM information_schema.columns
                                        WHERE table_name = :tbl
                                          AND column_name = :col
                                    """), {
                                        'tbl': table,
                                        'col': watermark_col
                                    }).fetchone(),
                                    f"validar_watermark_col_inicial_{db_name}.{table}"
                                )
                            if not col_ok:
                                print(f"  [INFO] '{watermark_col}' no existe en fuente -> se omite watermark")
                                watermark_col = ''

                    print(f"  modo={load_mode} | watermark_col={watermark_col or '-'} | watermark_val={watermark_val or '-'}")

                    full_refresh_prepared = False
                    if load_mode == 'full_refresh':
                        _prepare_full_refresh_target(
                            dst_engine=dst_engine,
                            src_engine=src_engine,
                            source_table=table,
                            target_table=target_table,
                        )
                        full_refresh_prepared = True

                    schema_changes = []

                    first = True
                    for chunk in pd.read_sql(query, src_engine, chunksize=CHUNK_SIZE):
                        chunk['_source_entity'] = source_entity
                        chunk['_extraction_id'] = item_extraction_id
                        chunk['_load_timestamp'] = datetime.now()

                        if first:
                            chunk_columns_with_types = _build_columns_with_types_from_chunk(chunk)
                            if run_with_retry(
                                lambda: _table_exists_in_bronze(dst_engine, target_table),
                                f"check_target_for_schema_sync_{db_name}.{table}->{target_table}"
                            ):
                                schema_changes = _sync_missing_columns_in_bronze(
                                    dst_engine=dst_engine,
                                    target_table=target_table,
                                    source_columns_with_types=chunk_columns_with_types,
                                )

                        if first and not full_refresh_prepared:
                            run_with_retry(
                                lambda: chunk.head(0).to_sql(
                                    name=target_table,
                                    con=dst_engine,
                                    schema='bronze',
                                    if_exists=if_exists_mode,
                                    index=False
                                ),
                                f"crear_o_reemplazar_tabla_{db_name}.{table}->{target_table}"
                            )

                            with dst_engine.begin() as ddl_conn:
                                run_with_retry(
                                    lambda: ddl_conn.execute(sa.text(
                                        f'ALTER TABLE bronze."{target_table}" '
                                        f'ADD COLUMN IF NOT EXISTS _row_id BIGSERIAL'
                                    )),
                                    f"agregar_row_id_{db_name}.{table}->{target_table}"
                                )
                        first = False

                        chunk = _coerce_integer_columns(chunk)

                        run_with_retry(
                            lambda: chunk.to_sql(
                                name=target_table,
                                con=dst_engine,
                                schema='bronze',
                                if_exists='append',
                                index=False,
                                method=psql_copy,
                                chunksize=CHUNK_SIZE
                            ),
                            f"cargar_chunk_{db_name}.{table}->{target_table}"
                        )
                        total_rows += len(chunk)

                    fin = datetime.now()

                    new_watermark_val = watermark_val
                    if watermark_col:
                        with src_engine.connect() as wm:
                            row = run_with_retry(
                                lambda: wm.execute(sa.text(
                                    f'SELECT MAX("{watermark_col}") FROM "{table}"'
                                )).fetchone(),
                                f"leer_max_watermark_{db_name}.{table}"
                            )
                            if row and row[0] is not None:
                                new_watermark_val = row[0]

                    with dst_engine.begin() as ctrl:
                        run_with_retry(
                            lambda: ctrl.execute(sa.text(f"""
                                INSERT INTO bronze._control_carga
                                    (db_name, table_name, target_table_name, extraction_id, filas_cargadas, fecha_carga, inicio, fin, status, load_mode, watermark_col, watermark_value)
                                VALUES
                                    (:db, :tbl, :target_tbl, :eid, :filas, :hoy, :inicio, :fin, 'EXPORTADO', :lm, :wc, :wv)
                                ON CONFLICT (db_name, table_name, fecha_carga) WHERE status = 'EXPORTADO'
                                DO UPDATE SET
                                    target_table_name = EXCLUDED.target_table_name,
                                    extraction_id = EXCLUDED.extraction_id,
                                    filas_cargadas = EXCLUDED.filas_cargadas,
                                    inicio = EXCLUDED.inicio,
                                    fin = EXCLUDED.fin,
                                    status = EXCLUDED.status,
                                    load_mode = EXCLUDED.load_mode,
                                    watermark_col = EXCLUDED.watermark_col,
                                    watermark_value = EXCLUDED.watermark_value
                            """), {
                                'db': db_name,
                                'tbl': table,
                                'target_tbl': target_table,
                                'eid': item_extraction_id,
                                'filas': total_rows,
                                'hoy': hoy,
                                'inicio': start,
                                'fin': fin,
                                'lm': load_mode,
                                'wc': watermark_col or None,
                                'wv': new_watermark_val
                            }),
                            f"insert_control_ok_{db_name}.{table}->{target_table}"
                        )

                    src_engine.dispose()

                    print(f"  OK {total_rows} filas | {(fin - start).seconds}s")
                    audit_summary.append({
                        'db': db_name,
                        'tabla': table,
                        'target_table': target_table,
                        'status': 'EXPORTADO',
                        'filas': total_rows,
                        'extraction_id': item_extraction_id,
                        'duracion_seg': (fin - start).seconds,
                        'load_mode': load_mode,
                        'watermark_col': watermark_col or None,
                        'watermark_val_inicio': watermark_val,
                        'schema_changes': schema_changes,
                    })

                except Exception as e:
                    print(f"  ERROR: {str(e)[:120]}")
                    with dst_engine.begin() as ctrl:
                        run_with_retry(
                            lambda: ctrl.execute(sa.text(f"""
                                INSERT INTO bronze._control_carga
                                    (db_name, table_name, target_table_name, extraction_id, fecha_carga, inicio, status, error_msg)
                                VALUES
                                    (:db, :tbl, :target_tbl, :eid, :hoy, :inicio, 'ERROR', :err)
                            """), {
                                'db': db_name,
                                'tbl': table,
                                'target_tbl': target_table,
                                'eid': item_extraction_id,
                                'hoy': hoy,
                                'inicio': start,
                                'err': str(e)
                            }),
                            f"insert_control_error_{db_name}.{table}->{target_table}"
                        )
                    audit_summary.append({
                        'db': db_name,
                        'tabla': table,
                        'target_table': target_table,
                        'status': 'ERROR_EXPORT',
                        'error': str(e)
                    })

                completados = sum(
                    1 for x in audit_summary
                    if x.get('status') in ('EXPORTADO', 'SALTADO_IDEMPOTENCIA', 'ERROR_EXPORT')
                )
                print(f"  >>> Progreso: {completados}/{total} tablas procesadas ({round(completados / total * 100)}%)")

            elif item.get('type') == 'file':
                file_info = item.get('file_info') or {}
                file_name = item.get('table_name')
                target_table = _target_tbl
                start = datetime.now()

                if not file_name or not target_table:
                    audit_summary.append({
                        'db': 'SMB_SERVER',
                        'tabla': file_name,
                        'target_table': target_table,
                        'status': 'ERROR_EXPORT',
                        'error': 'Item file sin table_name o target_table_name'
                    })
                    continue

                source_entity = file_info.get('source_entity', 'SMB_PATH')
                source_prefix = file_info.get('source_prefix', 'SMB')
                item_extraction_id = f"{source_prefix}-{run_ts}-{extraction_id_suffix}"

                try:
                    unc = smb_to_unc(file_info.get('path', ''), smb_host, smb_share)

                    with smbclient.open_file(unc, mode='rb') as f:
                        if file_info.get('type') == 'csv':
                            df = pd.read_csv(
                                f,
                                delimiter=file_info.get('delimiter', ','),
                                encoding=file_info.get('encoding', 'utf-8')
                            )
                        elif file_info.get('type') == 'excel':
                            df = pd.read_excel(f, sheet_name=file_info.get('sheet', 0))
                        else:
                            raise Exception(f"Tipo no soportado: {file_info.get('type')}")

                    df['_source_entity'] = source_entity
                    df['_extraction_id'] = item_extraction_id
                    df['_load_timestamp'] = datetime.now()

                    run_with_retry(
                        lambda: df.to_sql(
                            name=target_table,
                            con=dst_engine,
                            schema='bronze',
                            if_exists='replace',
                            index=False,
                            method=psql_copy,
                            chunksize=CHUNK_SIZE
                        ),
                        f"cargar_archivo_smb_{file_name}->{target_table}"
                    )

                    with dst_engine.begin() as ddl_conn:
                        run_with_retry(
                            lambda: ddl_conn.execute(sa.text(
                                f'ALTER TABLE bronze."{target_table}" '
                                f'ADD COLUMN IF NOT EXISTS _row_id BIGSERIAL'
                            )),
                            f"agregar_row_id_smb_{file_name}->{target_table}"
                        )

                    fin = datetime.now()

                    with dst_engine.begin() as ctrl:
                        run_with_retry(
                            lambda: ctrl.execute(sa.text(f"""
                                INSERT INTO bronze._control_carga
                                    (db_name, table_name, target_table_name, extraction_id, filas_cargadas, fecha_carga, inicio, fin, status)
                                VALUES
                                    (:db, :tbl, :target_tbl, :eid, :filas, :hoy, :inicio, :fin, 'EXPORTADO')
                                ON CONFLICT (db_name, table_name, fecha_carga) WHERE status = 'EXPORTADO'
                                DO UPDATE SET
                                    target_table_name = EXCLUDED.target_table_name,
                                    extraction_id = EXCLUDED.extraction_id,
                                    filas_cargadas = EXCLUDED.filas_cargadas,
                                    inicio = EXCLUDED.inicio,
                                    fin = EXCLUDED.fin,
                                    status = EXCLUDED.status
                            """), {
                                'db': 'SMB_SERVER',
                                'tbl': file_name,
                                'target_tbl': target_table,
                                'eid': item_extraction_id,
                                'filas': len(df),
                                'hoy': hoy,
                                'inicio': start,
                                'fin': fin
                            }),
                            f"insert_control_ok_smb_{file_name}->{target_table}"
                        )

                    audit_summary.append({
                        'db': 'SMB_SERVER',
                        'tabla': file_name,
                        'target_table': target_table,
                        'status': 'EXPORTADO',
                        'filas': len(df),
                        'extraction_id': item_extraction_id,
                        'duracion_seg': (fin - start).seconds
                    })

                except Exception as e:
                    with dst_engine.begin() as ctrl:
                        run_with_retry(
                            lambda: ctrl.execute(sa.text(f"""
                                INSERT INTO bronze._control_carga
                                    (db_name, table_name, target_table_name, extraction_id, fecha_carga, inicio, status, error_msg)
                                VALUES
                                    (:db, :tbl, :target_tbl, :eid, :hoy, :inicio, 'ERROR', :err)
                            """), {
                                'db': 'SMB_SERVER',
                                'tbl': file_name,
                                'target_tbl': target_table,
                                'eid': item_extraction_id,
                                'hoy': hoy,
                                'inicio': start,
                                'err': str(e)
                            }),
                            f"insert_control_error_smb_{file_name}->{target_table}"
                        )
                    audit_summary.append({
                        'db': 'SMB_SERVER',
                        'tabla': file_name,
                        'target_table': target_table,
                        'status': 'ERROR_EXPORT',
                        'error': str(e)
                    })

            else:
                audit_summary.append({
                    'db': item.get('db_name'),
                    'tabla': item.get('table_name'),
                    'target_table': _target_tbl,
                    'status': 'SKIPPED_TIPO_INVALIDO',
                    'error': f"Tipo no esperado: {item.get('type')}"
                })

    except Exception as exc:
        pipeline_fatal_error = exc
        pipeline_fatal_trace = traceback.format_exc()
        raise

    finally:
        if dst_engine is not None:
            try:
                log_rows = build_pipeline_log_rows(
                    manifest,
                    audit_summary,
                    pipeline_extraction_id,
                    fatal_error=pipeline_fatal_error,
                    fatal_trace=pipeline_fatal_trace,
                )
                insert_pipeline_logs(dst_engine, log_rows)
            except Exception as log_error:
                print(f"WARN: No se pudo registrar logs de pipeline: {log_error}")

            try:
                _write_finished_at = datetime.now()
                _timing_rows = []
                for tm in _timing_from_manifest:
                    st = datetime.fromisoformat(tm['started_at'])
                    ft = datetime.fromisoformat(tm['finished_at'])
                    _timing_rows.append({
                        'execution_id': pipeline_extraction_id,
                        'pipeline_name': 'bronze_landing_zone',
                        'activity_name': tm['activity'],
                        'started_at': st,
                        'finished_at': ft,
                        'duration_ms': int((ft - st).total_seconds() * 1000),
                        'status': 'SUCCESS',
                        'error_message': None,
                    })
                _timing_rows.append({
                    'execution_id': pipeline_extraction_id,
                    'pipeline_name': 'bronze_landing_zone',
                    'activity_name': 'bronze_write_exporter',
                    'started_at': _write_started_at,
                    'finished_at': _write_finished_at,
                    'duration_ms': int((_write_finished_at - _write_started_at).total_seconds() * 1000),
                    'status': 'ERROR' if pipeline_fatal_error else 'SUCCESS',
                    'error_message': str(pipeline_fatal_error)[:200] if pipeline_fatal_error else None,
                })
                insert_execution_times(dst_engine, _timing_rows)
            except Exception as te:
                print(f"WARN: No se pudo registrar tiempos de ejecucion: {te}")

            dst_engine.dispose()

    # Propaga un id de ejecucion a nivel run para trazabilidad consistente
    # entre writer, row_audit y email exporter.
    for x in audit_summary:
        if isinstance(x, dict) and not x.get('pipeline_execution_id'):
            x['pipeline_execution_id'] = pipeline_extraction_id

    exportados = sum(1 for x in audit_summary if x.get('status') == 'EXPORTADO')
    saltados = sum(1 for x in audit_summary if x.get('status') == 'SALTADO_IDEMPOTENCIA')
    print(f"Finalizado: {exportados} exportados | {saltados} saltados | {len(audit_summary)} total")
    return audit_summary