import sqlalchemy as sa
from urllib.parse import quote_plus
from mage_ai.data_preparation.shared.secrets import get_secret_value
from collections import Counter
from datetime import datetime
import json
import socket

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


def resolve_target_table_name(db_name, table_name, item_type, target_name=None):
    if target_name and str(target_name).strip():
        return str(target_name).strip()

    if item_type == 'file':
        return table_name

    return f"{db_name}_{table_name}"


def apply_target_table_validation(manifest):
    seen_targets = {}
    for item in manifest:
        if item.get('status') != 'READY':
            continue

        target_table_name = item.get('target_table_name')
        if not target_table_name:
            item['status'] = 'ERROR'
            item['error'] = 'Item sin target_table_name valido'
            continue

        source_key = (
            item.get('type'),
            item.get('db_name'),
            item.get('table_name'),
        )
        seen_targets.setdefault(target_table_name, []).append(source_key)

    duplicated_targets = {
        target: sources
        for target, sources in seen_targets.items()
        if len(sources) > 1
    }

    if not duplicated_targets:
        return

    for item in manifest:
        target_table_name = item.get('target_table_name')
        if item.get('status') != 'READY' or target_table_name not in duplicated_targets:
            continue

        item['status'] = 'ERROR'
        item['error'] = (
            f"target_table_name duplicado: {target_table_name}. "
            "Define target_name explicito en el JSON."
        )


def ensure_control_carga_columns(dst_host, dst_user, dst_pass):
    engine = None
    try:
        uri = (
            f"postgresql://{quote_plus(dst_user)}:"
            f"{quote_plus(dst_pass)}@"
            f"{dst_host}:5432/postgres"
        )
        engine = sa.create_engine(uri, connect_args={'client_encoding': 'utf8'})
        with engine.begin() as conn:
            conn.execute(sa.text("""
                ALTER TABLE bronze._control_carga
                ADD COLUMN IF NOT EXISTS target_table_name VARCHAR(200),
                ADD COLUMN IF NOT EXISTS watermark_col VARCHAR(128),
                ADD COLUMN IF NOT EXISTS watermark_value TIMESTAMP NULL,
                ADD COLUMN IF NOT EXISTS load_mode VARCHAR(30) DEFAULT 'full_refresh';
            """))
        print("OK: _control_carga validada con columnas target_table_name/watermark/load_mode.")
    except Exception as e:
        print(f"WARN: No se pudo validar _control_carga: {e}")
    finally:
        if engine is not None:
            engine.dispose()


@data_loader
def load_data(*args, **kwargs):
    _load_started_at = datetime.now()
    raw = kwargs.get('MIGRATION_TABLE')
    if not raw:
        print("Sin variable MIGRATION_TABLE - retornando manifiesto vacio.")
        return []

    config = json.loads(raw) if isinstance(raw, str) else raw

    host_db = get_secret_value('BDWORKFLOW_HOST')
    user_db = get_secret_value('BDWORKFLOW_USER')
    password_db = get_secret_value('BDWORKFLOW_PASSWORD')
    smb_host = get_secret_value('smb-server')

    dst_host = get_secret_value('BDMANAGER_HOST')
    dst_user = get_secret_value('BDMANAGER_USER')
    dst_pass = get_secret_value('BDMANAGER_PASSWORD')
    ensure_control_carga_columns(dst_host, dst_user, dst_pass)

    manifest = []

    def normalize_load_mode(value):
        mode = (value or 'full').strip().lower()
        return mode if mode in ('full', 'incremental') else 'full'

    def parse_tables_config(raw_tables):
        table_overrides = {}
        expand_all = False
        for item in raw_tables or []:
            if isinstance(item, str):
                if item.strip().upper() == 'ALL':
                    expand_all = True
                else:
                    table_overrides[item] = {}
            elif isinstance(item, dict):
                tname = item.get('name') or item.get('table_name') or item.get('table')
                if tname:
                    table_overrides[tname] = item
        return expand_all, table_overrides

    if 'databases' in config.get('sources', {}):
        for db_name, db_config in config['sources']['databases'].items():
            engine = None
            try:
                uri = (
                    f"postgresql://{quote_plus(user_db)}:"
                    f"{quote_plus(password_db)}@"
                    f"{host_db}:5432/{db_name}"
                )
                engine = sa.create_engine(
                    uri,
                    connect_args={'connect_timeout': 5, 'client_encoding': 'utf8'}
                )

                with engine.connect():
                    raw_tables = db_config.get('tables', [])
                    expand_all, table_overrides = parse_tables_config(raw_tables)

                    if expand_all:
                        tables = sa.inspect(engine).get_table_names()
                    else:
                        tables = list(table_overrides.keys())

                    for table in tables:
                        table_cfg = table_overrides.get(table, {})

                        load_mode = normalize_load_mode(
                            table_cfg.get('load_mode', db_config.get('load_mode', 'full'))
                        )
                        target_name = table_cfg.get('target_name', db_config.get('target_name'))
                        watermark_col = table_cfg.get('watermark_col', db_config.get('watermark_col'))
                        source_entity = table_cfg.get('source_entity', f"{db_name}.{table}")
                        source_prefix = table_cfg.get('source_prefix', db_config.get('source_prefix', db_name))
                        target_table_name = resolve_target_table_name(
                            db_name,
                            table,
                            'database',
                            target_name,
                        )

                        manifest.append({
                            'db_name': db_name,
                            'table_name': table,
                            'target_name': target_name,
                            'target_table_name': target_table_name,
                            'status': 'READY',
                            'type': 'database',
                            'load_mode': load_mode,
                            'watermark_col': watermark_col,
                            'source_entity': source_entity,
                            'source_prefix': source_prefix,
                        })

            except Exception as e:
                manifest.append({
                    'db_name': db_name,
                    'table_name': None,
                    'status': 'ERROR',
                    'error': str(e),
                    'type': 'database'
                })
            finally:
                if engine is not None:
                    engine.dispose()

    if 'files' in config.get('sources', {}):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            smb_ok = s.connect_ex((smb_host, 445)) == 0
            s.close()
        except Exception:
            smb_ok = False

        for file_info in config['sources']['files']:
            load_mode = normalize_load_mode(file_info.get('load_mode', 'full'))
            target_name = file_info.get('target_name')
            watermark_col = file_info.get('watermark_col')
            source_entity = file_info.get('source_entity', file_info.get('name'))
            source_prefix = file_info.get('source_prefix', file_info.get('prefix', 'SMB'))
            target_table_name = resolve_target_table_name(
                'SMB_SERVER',
                file_info.get('name'),
                'file',
                target_name,
            )

            manifest.append({
                'db_name': 'SMB_SERVER',
                'table_name': file_info.get('name'),
                'target_name': target_name,
                'target_table_name': target_table_name,
                'file_info': file_info,
                'status': 'READY' if smb_ok else 'ERROR',
                'error': None if smb_ok else 'Puerto 445 inaccesible',
                'type': 'file',
                'load_mode': load_mode,
                'watermark_col': watermark_col,
                'source_entity': source_entity,
                'source_prefix': source_prefix,
            })

    # Validación adicional para target_table_name en tablas nuevas
    for item in manifest:
        if item.get('status') == 'READY' and not item.get('target_table_name'):
            item['status'] = 'ERROR'
            item['error'] = 'Falta target_table_name en tabla nueva'

    apply_target_table_validation(manifest)

    print(f"Total items en manifest: {len(manifest)}")
    print(f"Por status: {dict(Counter(x.get('status') for x in manifest))}")
    print(f"Por tipo: {dict(Counter(x.get('type') for x in manifest))}")
    print(f"Primeros 3: {manifest[:3]}")

    errores = [x for x in manifest if x.get('status') == 'ERROR']
    print(f"Total errores: {len(errores)}")
    for e in errores:
        print({
            'db_name': e.get('db_name'),
            'table_name': e.get('table_name'),
            'error': e.get('error')
        })

    _load_finished_at = datetime.now()
    manifest.append({
        '_timing_meta': True,
        'activity': 'bronze_load',
        'started_at': _load_started_at.isoformat(),
        'finished_at': _load_finished_at.isoformat(),
    })

    return manifest