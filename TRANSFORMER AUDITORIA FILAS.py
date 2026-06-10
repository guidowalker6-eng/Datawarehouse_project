import re
from collections import Counter
from datetime import datetime
from urllib.parse import quote_plus

import sqlalchemy as sa
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_AUDITABLE_STATUSES = {'EXPORTADO', 'SALTADO_IDEMPOTENCIA'}


def _log_row_audit(step, **kwargs):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload = ' | '.join(f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())) if kwargs else ''
    print(f"[ROW_AUDIT][{ts}] {step}" + (f" | {payload}" if payload else ''))


def _is_valid_identifier(value):
    return bool(value and _IDENTIFIER_RE.match(str(value)))


def _sanitize_for_mage(value):
    """Evita problemas de serializacion en Mage con valores None/objetos complejos."""
    if value is None:
        return ''
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_sanitize_for_mage(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_for_mage(v) for k, v in value.items()}
    return str(value)


def _mark_row_audit_block_error(rows, error_msg):
    out = []
    message = str(error_msg or '')[:220]
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row.setdefault('audit_source_rows', None)
        row.setdefault('audit_target_rows', None)
        row.setdefault('audit_status', 'NO_AUDITABLE')
        row.setdefault('audit_error_msg', message or 'Fallo interno en row audit transformer')
        row.setdefault('source_count', None)
        row.setdefault('dw_count', None)
        row.setdefault('difference', None)
        row.setdefault('audit_reason', message or 'Fallo interno en row audit transformer')
        row['row_audit_block_status'] = 'ERROR'
        row['row_audit_block_error'] = message or 'Fallo interno en row audit transformer'
        out.append(row)
    return out


def _pick_audit_input(*args):
    """Selecciona el input mas probable de export_escritura cuando hay multiples upstream."""
    candidates = []
    for idx, arg in enumerate(args):
        data = None
        if isinstance(arg, list):
            data = arg
        elif isinstance(arg, dict):
            data = [arg]
        if data is None:
            continue

        auditables = sum(
            1 for x in data
            if isinstance(x, dict) and str(x.get('status') or '').strip().upper() in _AUDITABLE_STATUSES
        )
        candidates.append((auditables, idx, data))

    if not candidates:
        return []

    # Prioriza el candidato con mayor cantidad de estados auditables.
    candidates.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    _log_row_audit(
        'pick_input',
        candidates=len(candidates),
        selected_index=candidates[0][1],
        selected_auditables=candidates[0][0],
        selected_total=len(candidates[0][2]),
    )
    return candidates[0][2]


def _normalize_load_mode(value):
    mode = str(value or '').strip().lower().replace(' ', '_')
    if mode in ('full', 'full_refresh'):
        return 'full_refresh'
    if mode == 'incremental':
        return 'incremental'
    return 'full_refresh'


def _build_db_engine(host, user, password, db_name):
    return sa.create_engine(
        f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:5432/{db_name}",
        connect_args={'connect_timeout': 10, 'client_encoding': 'utf8'},
    )


def _resolve_row_id_column(dst_engine, target_table):
    with dst_engine.connect() as conn:
        rows = conn.execute(sa.text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'bronze'
              AND table_name = :tbl
              AND column_name IN ('row_id', '_row_id')
            ORDER BY CASE WHEN column_name = 'row_id' THEN 0 ELSE 1 END
        """), {'tbl': target_table}).fetchall()
    return rows[0][0] if rows else None


def _source_count_full_refresh_fast(src_engine, source_table):
    # Conteo estimado rapido por estadisticas de postgres para evitar full scan.
    relname_candidates = [source_table, f'public.{source_table}']

    with src_engine.connect() as conn:
        for relname in relname_candidates:
            row = conn.execute(sa.text("""
                SELECT c.reltuples::bigint
                FROM pg_class c
                WHERE c.oid = to_regclass(:regname)
            """), {'regname': relname}).fetchone()
            if row and row[0] is not None:
                return int(row[0])

    return None


def _source_count_full_refresh_exact(src_engine, source_table):
    with src_engine.connect() as conn:
        row = conn.execute(sa.text(f'SELECT COUNT(*) FROM "{source_table}"')).fetchone()
    return int(row[0] if row else 0)


def _count_incremental(src_engine, source_table, watermark_col, watermark_value):
    with src_engine.connect() as conn:
        row = conn.execute(sa.text(
            f'SELECT COUNT(*) FROM "{source_table}" WHERE "{watermark_col}" > :wv'
        ), {'wv': watermark_value}).fetchone()
    return int(row[0] if row else 0)


def _count_target_full(dst_engine, target_table, row_id_col=None):
    query = f'SELECT COUNT(*) FROM bronze."{target_table}"'
    if row_id_col:
        query += f' WHERE "{row_id_col}" >= 0'
    with dst_engine.connect() as conn:
        row = conn.execute(sa.text(query)).fetchone()
    return int(row[0] if row else 0)


def _count_target_incremental(dst_engine, target_table, watermark_col, watermark_value, row_id_col=None):
    where = [f'"{watermark_col}" > :wv']
    if row_id_col:
        where.append(f'"{row_id_col}" >= 0')
    query = f'SELECT COUNT(*) FROM bronze."{target_table}" WHERE ' + ' AND '.join(where)
    with dst_engine.connect() as conn:
        row = conn.execute(sa.text(query), {'wv': watermark_value}).fetchone()
    return int(row[0] if row else 0)


def _write_audit_timing(dst_host, dst_user, dst_pass, execution_id, started_at, finished_at, error_msg=None):
    engine = None
    try:
        engine = _build_db_engine(dst_host, dst_user, dst_pass, 'postgres')
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)
        with engine.begin() as conn:
            conn.execute(sa.text("""
                INSERT INTO bronze._pipeline_execution_times
                    (execution_id, pipeline_name, activity_name, started_at, finished_at,
                     duration_ms, status, error_message)
                VALUES
                    (:execution_id, :pipeline_name, :activity_name, :started_at, :finished_at,
                     :duration_ms, :status, :error_message)
            """), {
                'execution_id': execution_id,
                'pipeline_name': 'bronze_landing_zone',
                'activity_name': 'bronze_row_audit_transformer',
                'started_at': started_at,
                'finished_at': finished_at,
                'duration_ms': duration_ms,
                'status': 'ERROR' if error_msg else 'SUCCESS',
                'error_message': error_msg,
            })
    except Exception as te:
        print(f"WARN: No se pudo registrar tiempo de auditoria: {te}")
    finally:
        if engine:
            engine.dispose()


@transformer
def transform(audit_summary, *args, **kwargs):
    started_at = datetime.now()
    transformer_error = None
    _log_row_audit('transform_start', input_type=type(audit_summary).__name__)

    if audit_summary is None:
        return []
    if isinstance(audit_summary, dict):
        audit_summary = [audit_summary]
    if not isinstance(audit_summary, list):
        _log_row_audit('transform_end_early', reason='input_not_list')
        return []

    _log_row_audit('input_normalized', total_items=len(audit_summary))

    src_host = src_user = src_pass = None
    dst_host = dst_user = dst_pass = None

    out = []
    src_engines = {}
    dst_engine = None

    try:
        src_host = get_secret_value('BDWORKFLOW_HOST')
        src_user = get_secret_value('BDWORKFLOW_USER')
        src_pass = get_secret_value('BDWORKFLOW_PASSWORD')

        dst_host = get_secret_value('BDMANAGER_HOST')
        dst_user = get_secret_value('BDMANAGER_USER')
        dst_pass = get_secret_value('BDMANAGER_PASSWORD')
        _log_row_audit('secrets_loaded', src_host=bool(src_host), dst_host=bool(dst_host))

        dst_engine = _build_db_engine(dst_host, dst_user, dst_pass, 'postgres')
        _log_row_audit('dst_engine_ready')

        for idx, item in enumerate(audit_summary, start=1):
            if not isinstance(item, dict):
                _log_row_audit('skip_non_dict_item', index=idx)
                continue

            row = dict(item)

            # Se audita lo exportado y lo saltado por idempotencia.
            row_status = str(row.get('status') or '').strip().upper()
            _log_row_audit(
                'row_start',
                index=idx,
                status=row_status,
                db=row.get('db'),
                source_table=row.get('tabla'),
                target_table=row.get('target_table'),
            )
            if row_status not in _AUDITABLE_STATUSES:
                row.setdefault('audit_source_rows', None)
                row.setdefault('audit_target_rows', None)
                row.setdefault('audit_status', 'NO_AUDITABLE')
                row.setdefault('audit_error_msg', 'No aplica: estado no auditable')
                _log_row_audit('row_skipped_not_auditable', index=idx, status=row_status)
                out.append(row)
                continue

            db_name = row.get('db')
            source_table = row.get('tabla')
            target_table = row.get('target_table')

            if db_name == 'SMB_SERVER':
                row.update({
                    'audit_source_rows': None,
                    'audit_target_rows': None,
                    'audit_status': 'NO_AUDITABLE',
                    'audit_error_msg': 'Fuente tipo archivo SMB',
                })
                _log_row_audit('row_no_auditable_smb', index=idx)
                out.append(row)
                continue

            if not (_is_valid_identifier(source_table) and _is_valid_identifier(target_table) and _is_valid_identifier(db_name)):
                row.update({
                    'audit_source_rows': None,
                    'audit_target_rows': None,
                    'audit_status': 'FAILED',
                    'audit_error_msg': 'Identificador invalido en db/tabla origen o destino',
                })
                _log_row_audit('row_failed_invalid_identifier', index=idx, db=db_name, source_table=source_table, target_table=target_table)
                out.append(row)
                continue

            load_mode = _normalize_load_mode(row.get('load_mode'))
            watermark_col = row.get('watermark_col')
            watermark_value = (
                row.get('lwatermark_value')
                or row.get('watermark_value')
                or row.get('watermark_val_inicio')
            )
            _log_row_audit(
                'row_mode_resolved',
                index=idx,
                load_mode=load_mode,
                watermark_col=watermark_col,
                has_watermark_value=watermark_value is not None,
            )

            try:
                if db_name not in src_engines:
                    src_engines[db_name] = _build_db_engine(src_host, src_user, src_pass, db_name)
                    _log_row_audit('src_engine_created', index=idx, db=db_name)
                src_engine = src_engines[db_name]

                row_id_col = _resolve_row_id_column(dst_engine, target_table)
                _log_row_audit('row_id_column_resolved', index=idx, row_id_col=row_id_col or '-')

                if load_mode == 'incremental' and _is_valid_identifier(watermark_col) and watermark_value is not None:
                    _log_row_audit('count_strategy', index=idx, strategy='incremental')
                    source_count = _count_incremental(src_engine, source_table, watermark_col, watermark_value)
                    target_count = _count_target_incremental(
                        dst_engine, target_table, watermark_col, watermark_value, row_id_col
                    )
                else:
                    _log_row_audit('count_strategy', index=idx, strategy='full_refresh_fast')
                    source_count = _source_count_full_refresh_fast(src_engine, source_table)
                    if source_count is None:
                        _log_row_audit('count_strategy_fallback', index=idx, strategy='full_refresh_exact')
                        source_count = _source_count_full_refresh_exact(src_engine, source_table)
                    target_count = _count_target_full(dst_engine, target_table, row_id_col)

                diff = int(source_count) - int(target_count)
                if diff == 0:
                    audit_status = 'SUCCESS'
                    audit_error_msg = None
                else:
                    audit_status = 'FAILED'
                    audit_error_msg = f'Descuadre de {diff} filas (origen={source_count}, dw={target_count})'

                row.update({
                    'audit_source_rows': int(source_count),
                    'audit_target_rows': int(target_count),
                    'audit_status': audit_status,
                    'audit_error_msg': audit_error_msg,
                    # Compatibilidad con formato usado en email exporter.
                    'source_count': int(source_count),
                    'dw_count': int(target_count),
                    'difference': diff,
                    'audit_reason': audit_error_msg,
                    'row_audit_block_status': 'SUCCESS',
                    'row_audit_block_error': None,
                })
                _log_row_audit(
                    'row_audit_done',
                    index=idx,
                    source_count=source_count,
                    target_count=target_count,
                    difference=diff,
                    audit_status=audit_status,
                )

            except Exception as exc:
                row.update({
                    'audit_source_rows': None,
                    'audit_target_rows': None,
                    'audit_status': 'FAILED',
                    'audit_error_msg': str(exc)[:220],
                    'source_count': None,
                    'dw_count': None,
                    'difference': None,
                    'audit_reason': str(exc)[:220],
                    'row_audit_block_status': 'SUCCESS',
                    'row_audit_block_error': None,
                })
                _log_row_audit('row_audit_error', index=idx, error=str(exc)[:180])

            out.append(row)

    except Exception as fatal:
        transformer_error = str(fatal)[:220]
        print(f"WARN: Fallo global en auditoria de filas. Se retorna input sin romper pipeline. error={fatal}")
        _log_row_audit('transform_fatal', error=transformer_error)
        out = _mark_row_audit_block_error(audit_summary, fatal)

    finally:
        for engine in src_engines.values():
            try:
                engine.dispose()
            except Exception:
                pass
        if dst_engine is not None:
            try:
                dst_engine.dispose()
            except Exception:
                pass

        finished_at = datetime.now()
        execution_id = next(
            (
                x.get('pipeline_execution_id')
                or x.get('execution_id')
                or x.get('extraction_id')
                for x in out
                if isinstance(x, dict)
                and (x.get('pipeline_execution_id') or x.get('execution_id') or x.get('extraction_id'))
            ),
            'SIN_ID',
        )
        if dst_host and dst_user and dst_pass:
            _write_audit_timing(dst_host, dst_user, dst_pass, execution_id, started_at, finished_at, transformer_error)
            _log_row_audit('timing_written', execution_id=execution_id, has_error=bool(transformer_error))

    audited = sum(1 for x in out if isinstance(x, dict) and x.get('audit_status') in ('SUCCESS', 'FAILED'))
    failed = sum(1 for x in out if isinstance(x, dict) and x.get('audit_status') == 'FAILED')
    if audited == 0:
        status_dist = Counter(
            str(x.get('status') or '').upper()
            for x in out
            if isinstance(x, dict)
        )
        print(f"row_audit_transformer: sin filas auditadas. status_dist={dict(status_dist)}")
    print(f"row_audit_transformer: auditadas={audited} failed={failed} total={len(out)}")
    _log_row_audit('transform_end', audited=audited, failed=failed, total=len(out))

    return _sanitize_for_mage(out)


@custom
def custom_block(*args, **kwargs):
    """Compatibilidad cuando el bloque en Mage esta tipado como Custom."""
    try:
        _log_row_audit('custom_block_start', args_count=len(args))
        selected_input = _pick_audit_input(*args)
        _log_row_audit('custom_block_selected_input', selected_items=len(selected_input))
        return transform(selected_input, **kwargs)
    except Exception as exc:
        print(f"WARN: custom_block row_audit fallo controlado: {exc}")
        _log_row_audit('custom_block_error', error=str(exc)[:180])
        safe_input = _pick_audit_input(*args)
        return _sanitize_for_mage(_mark_row_audit_block_error(safe_input, exc))
