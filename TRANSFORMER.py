import uuid
from datetime import datetime

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(manifest, *args, **kwargs):
    _transformer_started_at = datetime.now()
    # Normalizacion defensiva de entrada
    if manifest is None:
        return []
    if isinstance(manifest, dict):
        manifest = [manifest]
    if not isinstance(manifest, list):
        return []
    if len(manifest) == 0:
        return []

    exec_date = kwargs.get('execution_date') or datetime.now()
    ts_str = exec_date.strftime("%Y%m%d-%H%M")

    # Unico extraction_id del run:
    # si ya viene, se reutiliza; si no, se genera aqui una sola vez.
    extraction_id = (
        kwargs.get('extraction_id')
        or kwargs.get('EXTRACTION_ID')
        or f"EVO-{ts_str}-{str(uuid.uuid4())[:8].upper()}"
    )

    load_ts = datetime.now().isoformat()

    out = []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        if not enriched.get('target_table_name') and enriched.get('table_name'):
            target_name = enriched.get('target_name')
            item_type = enriched.get('type')
            db_name = enriched.get('db_name')
            table_name = enriched.get('table_name')
            if target_name and str(target_name).strip():
                enriched['target_table_name'] = str(target_name).strip()
            elif item_type == 'file':
                enriched['target_table_name'] = table_name
            elif db_name:
                enriched['target_table_name'] = f"{db_name}_{table_name}"
        enriched['extraction_id'] = extraction_id
        enriched['extraction_timestamp'] = load_ts
        out.append(enriched)

    print(f"transform: input={len(manifest)} output={len(out)} extraction_id={extraction_id}")

    _transformer_finished_at = datetime.now()
    out.append({
        '_timing_meta': True,
        'activity': 'bronze_transformer',
        'started_at': _transformer_started_at.isoformat(),
        'finished_at': _transformer_finished_at.isoformat(),
        'extraction_id': extraction_id,
    })

    return out