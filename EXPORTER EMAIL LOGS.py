import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from jinja2 import Template
from datetime import datetime
from mage_ai.data_preparation.shared.secrets import get_secret_value
import os
import io
import time
import traceback
import concurrent.futures
import pandas as pd
import sqlalchemy as sa
from urllib.parse import quote_plus

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


AUDITABLE_STATUSES = {'EXPORTADO', 'SALTADO_IDEMPOTENCIA'}


def _resolve_template_path():
    current_file = globals().get('__file__')
    current_dir = os.path.dirname(os.path.abspath(current_file)) if current_file else None
    parent_dir = os.path.dirname(current_dir) if current_dir else None

    candidate_dirs = []
    if current_dir:
        candidate_dirs.append(current_dir)
    if parent_dir:
        candidate_dirs.extend([
            os.path.join(parent_dir, "data_exporters"),
            parent_dir,
        ])

    for env_key in ('MAGE_REPO_PATH', 'MAGE_PROJECT_ROOT'):
        env_dir = os.getenv(env_key)
        if env_dir:
            candidate_dirs.extend([
                env_dir,
                os.path.join(env_dir, 'data_exporters'),
            ])

    candidate_dirs.extend([
        os.path.join(os.getcwd(), 'data_exporters'),
        os.getcwd(),
    ])

    # Preserve order while removing duplicates.
    unique_dirs = []
    seen = set()
    for directory in candidate_dirs:
        if directory and directory not in seen:
            seen.add(directory)
            unique_dirs.append(directory)

    candidates = [os.path.join(directory, 'email_template.html') for directory in unique_dirs]

    for path in candidates:
        if os.path.exists(path):
            return path

    checked = "\n - ".join(candidates)
    raise FileNotFoundError(
        "No se encontro email_template.html. Rutas verificadas:\n - " + checked
    )

def render_email_html(data):
    try:
        template_path = _resolve_template_path()
        with open(template_path, "r", encoding="utf-8") as file:
            template = Template(file.read())
    except Exception as error:
        print(f"WARN: No se pudo resolver email_template.html, se usa plantilla inline. error={error}")
        template = Template("""
        <html>
            <body style=\"font-family: Arial, sans-serif; color: #222;\">
                <h2>Resumen de ejecucion: {{ pipeline_name }}</h2>
                <p><strong>Estado:</strong> {{ status }}</p>
                <p><strong>Ambiente:</strong> {{ environment }}</p>
                <p><strong>Run ID:</strong> {{ run_id }}</p>
                <p><strong>Runtime:</strong> {{ runtime }}</p>
                <p><strong>Total registros:</strong> {{ total_records }}</p>
                <p><strong>Success rate:</strong> {{ success_rate }}%</p>
                <p><strong>Fecha:</strong> {{ datetime }}</p>
            </body>
        </html>
        """)
    return template.render(data)

def classify_severity(table_name):
    critical_tables = ["important_table_1", "important_table_2"]
    return "High" if table_name in critical_tables else "Low"

def suggest_action(error_type):
    error_text = str(error_type or "")
    if "ConnectionError" in error_text:
        return "Retry"
    elif "DataError" in error_text:
        return "Check DDL"
    else:
        return "Investigate"

def _required_secret(name):
    value = get_secret_value(name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Falta el secreto requerido: {name}")
    return str(value).strip()

def _get_smtp_port():
    raw = get_secret_value('smtp-port')
    if raw is None or str(raw).strip() == "":
        raw = get_secret_value('smtp_port')

    if raw is None or str(raw).strip() == "":
        print("INFO: smtp-port/smtp_port no definido, usando 587 por defecto")
        return 587

    try:
        return int(str(raw).strip())
    except Exception:
        raise ValueError(f"El secreto SMTP port no es numerico: {raw}")


def _parse_recipients(raw_value):
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if text == "":
        return []

    # Acepta correos separados por coma, punto y coma o salto de linea.
    normalized = text.replace("\n", ",").replace(";", ",")
    recipients = [item.strip() for item in normalized.split(",") if item.strip()]
    return recipients


def _get_test_recipient(kwargs):
    raw_value = (
        kwargs.get('EMAIL_SMTP_DEV')
        or kwargs.get('email_smtp_dev')
        or get_secret_value('EMAIL_SMTP_DEV')
        or get_secret_value('email_smtp_dev')
    )
    recipients = _parse_recipients(raw_value)
    if not recipients:
        raise ValueError(
            "No hay destinatario valido de pruebas. Define EMAIL_SMTP_DEV con uno o mas correos."
        )
    # En pruebas solo se envia a un unico destinatario (EMAIL_SMTP_DEV).
    recipient = recipients[0]
    if len(recipients) > 1:
        print("WARN: EMAIL_SMTP_DEV contiene multiples correos; se usara solo el primero para pruebas.")

    print(f"INFO: Modo pruebas email activo. Destinatario unico: {recipient}")
    return [recipient]


def _get_prod_recipients(kwargs):
    raw_value = (
        kwargs.get('EMAIL_SMTP_PROUD')
        or kwargs.get('email_smtp_proud')
        or get_secret_value('EMAIL_SMTP_PROUD')
        or get_secret_value('email_smtp_proud')
        # Fallback por si la variable esta registrada como PROD.
        or kwargs.get('EMAIL_SMTP_PROD')
        or kwargs.get('email_smtp_prod')
        or get_secret_value('EMAIL_SMTP_PROD')
        or get_secret_value('email_smtp_prod')
    )
    recipients = _parse_recipients(raw_value)
    if not recipients:
        raise ValueError(
            "No hay destinatarios validos. Define EMAIL_SMTP_PROUD con uno o mas correos."
        )

    print(f"INFO: Modo productivo email activo. Destinatarios: {', '.join(recipients)}")
    return recipients


def _is_retryable_smtp_error(error):
    if isinstance(error, (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError)):
        return True

    if isinstance(error, (TimeoutError, OSError)):
        return True

    if isinstance(error, smtplib.SMTPResponseException):
        code = int(getattr(error, 'smtp_code', 0) or 0)
        # 4xx suele representar errores transitorios del servidor SMTP.
        return 400 <= code < 500

    return False


def _build_message(subject, from_addr, recipients, html_body):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = ', '.join(recipients)
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))
    return msg


def _build_message_with_attachments(subject, from_addr, recipients, html_body, attachments=None):
    """Construye mensaje MIME con HTML + adjuntos opcionales."""
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = ', '.join(recipients)

    alt_part = MIMEMultipart('alternative')
    alt_part.attach(MIMEText(html_body, 'html', 'utf-8'))
    msg.attach(alt_part)

    for filename, data in (attachments or []):
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(part)

    return msg


def _send_email_with_retry(smtp_host, smtp_port, smtp_user, smtp_pass, recipients, msg,
                           max_attempts=3, base_delay_seconds=2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())
            print(f"INFO: Correo enviado en intento {attempt}/{max_attempts}")
            return
        except Exception as error:
            last_error = error
            retryable = _is_retryable_smtp_error(error)
            print(
                f"WARN: Fallo envio email intento {attempt}/{max_attempts}. "
                f"retryable={retryable}. error={error}"
            )
            if (not retryable) or attempt == max_attempts:
                break

            delay = base_delay_seconds * (2 ** (attempt - 1))
            print(f"INFO: Reintentando envio de correo en {delay}s")
            time.sleep(delay)

    raise last_error


def _send_failure_notification(smtp_host, smtp_port, smtp_user, smtp_pass, recipients,
                               pipeline_name, environment, run_id, original_error):
    error_trace = traceback.format_exc()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    subject = f"[{environment}] {pipeline_name} - EXPORTER_FAILURE - run_id={run_id}"
    body = f"""
    <html>
      <body style=\"font-family: Arial, sans-serif; color: #222;\">
        <h2 style=\"color:#b00020;\">Fallo en envio de logs del pipeline</h2>
        <p><strong>Pipeline:</strong> {pipeline_name}</p>
        <p><strong>Ambiente:</strong> {environment}</p>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Fecha:</strong> {now}</p>
        <p><strong>Error:</strong> {str(original_error)}</p>
        <h3>Traceback</h3>
        <pre style=\"background:#f6f8fa;padding:12px;border-radius:6px;white-space:pre-wrap;\">{error_trace}</pre>
      </body>
    </html>
    """

    msg = _build_message(subject, smtp_user, recipients, body)
    _send_email_with_retry(
        smtp_host,
        smtp_port,
        smtp_user,
        smtp_pass,
        recipients,
        msg,
        max_attempts=2,
        base_delay_seconds=1,
    )


# ---------------------------------------------------------------------------
# Auditoría de filas: conteo origen vs DW
# ---------------------------------------------------------------------------

def _audit_row_count_single(item, src_host, src_user, src_pass, dst_host, dst_user, dst_pass):
    """Ejecuta el conteo para UNA tabla según su load_mode. Thread-safe."""
    db_name      = item.get('db')
    source_table = item.get('tabla')
    target_table = item.get('target_table')
    load_mode    = item.get('load_mode', 'full_refresh')
    watermark_col       = item.get('watermark_col')
    watermark_val_inicio = item.get('watermark_val_inicio')

    if db_name == 'SMB_SERVER':
        return {**item, 'source_count': None, 'dw_count': None,
                'difference': None, 'audit_status': 'NO_AUDITABLE',
                'audit_reason': 'Fuente tipo archivo SMB'}

    src_engine = dst_engine = None
    try:
        src_engine = sa.create_engine(
            f"postgresql://{quote_plus(src_user)}:{quote_plus(src_pass)}"
            f"@{src_host}:5432/{db_name}",
            connect_args={'connect_timeout': 15, 'client_encoding': 'utf8'}
        )
        dst_engine = sa.create_engine(
            f"postgresql://{quote_plus(dst_user)}:{quote_plus(dst_pass)}"
            f"@{dst_host}:5432/postgres",
            connect_args={'connect_timeout': 15, 'client_encoding': 'utf8'}
        )

        if load_mode == 'full_refresh':
            with src_engine.connect() as c:
                source_count = c.execute(
                    sa.text(f'SELECT COUNT(*) FROM "{source_table}"')
                ).scalar()
            with dst_engine.connect() as c:
                dw_count = c.execute(
                    sa.text(f'SELECT COUNT(*) FROM bronze."{target_table}"')
                ).scalar()
        else:
            # incremental: solo filas nuevas usando la misma ventana que usó la carga
            with src_engine.connect() as c:
                source_count = c.execute(
                    sa.text(f'SELECT COUNT(*) FROM "{source_table}" WHERE "{watermark_col}" > :wv'),
                    {'wv': watermark_val_inicio}
                ).scalar()
            with dst_engine.connect() as c:
                dw_count = c.execute(
                    sa.text(f'SELECT COUNT(*) FROM bronze."{target_table}" WHERE "{watermark_col}" > :wv'),
                    {'wv': watermark_val_inicio}
                ).scalar()

        difference   = (source_count or 0) - (dw_count or 0)
        audit_status = 'OK' if source_count == dw_count else 'DISCREPANCIA'
        return {**item, 'source_count': source_count, 'dw_count': dw_count,
                'difference': difference, 'audit_status': audit_status, 'audit_reason': None}

    except Exception as exc:
        return {**item, 'source_count': None, 'dw_count': None,
                'difference': None, 'audit_status': 'ERROR_AUDITORIA',
                'audit_reason': str(exc)[:200]}
    finally:
        if src_engine:
            src_engine.dispose()
        if dst_engine:
            dst_engine.dispose()


def _build_row_audit(audit_summary, src_host, src_user, src_pass, dst_host, dst_user, dst_pass):
    """Ejecuta auditoría de filas en paralelo (max 4 workers) para tablas EXPORTADO."""
    auditables = [x for x in audit_summary if str(x.get('status') or '').upper() in AUDITABLE_STATUSES]
    if not auditables:
        return []

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                _audit_row_count_single, item,
                src_host, src_user, src_pass,
                dst_host, dst_user, dst_pass
            ): item
            for item in auditables
        }
        for future in concurrent.futures.as_completed(futures, timeout=180):
            try:
                results.append(future.result())
            except Exception as exc:
                item = futures[future]
                results.append({**item, 'source_count': None, 'dw_count': None,
                                 'difference': None, 'audit_status': 'ERROR_AUDITORIA',
                                 'audit_reason': str(exc)[:200]})
    return results


def _build_row_audit_from_precomputed(audit_summary):
    """Usa auditoria ya calculada en un transformer previo para evitar recontar en email."""
    auditables = [x for x in audit_summary if str(x.get('status') or '').upper() in AUDITABLE_STATUSES]
    if not auditables:
        return []

    # Requiere llaves de auditoria presentes para considerar que esta precomputado.
    precomputed = [
        x for x in auditables
        if x.get('audit_status') is not None
        and ('audit_source_rows' in x or 'source_count' in x)
        and ('audit_target_rows' in x or 'dw_count' in x)
    ]
    if len(precomputed) == 0:
        return []

    results = []
    for item in auditables:
        src = item.get('audit_source_rows', item.get('source_count'))
        dst = item.get('audit_target_rows', item.get('dw_count'))
        status = str(item.get('audit_status') or '').upper()
        reason = item.get('audit_error_msg') or item.get('audit_reason')

        if status == 'SUCCESS':
            audit_status = 'OK'
        elif status == 'FAILED':
            audit_status = 'DISCREPANCIA'
        else:
            audit_status = 'NO_AUDITABLE'

        difference = None
        if src is not None and dst is not None:
            try:
                difference = int(src) - int(dst)
            except Exception:
                difference = None

        results.append({
            **item,
            'source_count': src,
            'dw_count': dst,
            'difference': difference,
            'audit_status': audit_status,
            'audit_reason': reason,
        })

    return results


# ---------------------------------------------------------------------------
# Generación de Excel
# ---------------------------------------------------------------------------

def _generate_excel_export_audit(audit_summary):
    """Excel 1: log operativo actual de exportación."""
    rows = [{
        'Base de Datos':   r.get('db', ''),
        'Tabla Origen':    r.get('tabla', ''),
        'Tabla DW':        r.get('target_table', ''),
        'Estado':          r.get('status', ''),
        'Filas Cargadas':  r.get('filas', ''),
        'Modo Carga':      r.get('load_mode', ''),
        'Duracion (seg)':  r.get('duracion_seg', ''),
        'Extraction ID':   r.get('extraction_id', ''),
        'Error':           r.get('error', ''),
    } for r in audit_summary]

    buf = io.BytesIO()
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Export Audit')
    buf.seek(0)
    return buf.read()


def _generate_excel_row_audit(row_audit_results):
    """Excel 2: auditoría de filas origen vs DW (3 hojas)."""
    ok_rows      = [r for r in row_audit_results if r.get('audit_status') == 'OK']
    disc_rows    = [r for r in row_audit_results if r.get('audit_status') == 'DISCREPANCIA']
    noaud_rows   = [r for r in row_audit_results if r.get('audit_status') not in ('OK', 'DISCREPANCIA')]

    def _to_df(rows):
        return pd.DataFrame([{
            'Base de Datos':    r.get('db', ''),
            'Tabla Origen':     r.get('tabla', ''),
            'Tabla DW':         r.get('target_table', ''),
            'Modo Carga':       r.get('load_mode', ''),
            'Filas Origen':     r.get('source_count', ''),
            'Filas DW':         r.get('dw_count', ''),
            'Diferencia':       r.get('difference', ''),
            'Estado Auditoria': r.get('audit_status', ''),
            'Detalle':          r.get('audit_reason', ''),
        } for r in rows])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        _to_df(ok_rows).to_excel(writer,    index=False, sheet_name='Coincidencias')
        _to_df(disc_rows).to_excel(writer,  index=False, sheet_name='Discrepancias')
        _to_df(noaud_rows).to_excel(writer, index=False, sheet_name='No Auditables')
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Registro de tiempo del email en la tabla de tiempos
# ---------------------------------------------------------------------------

def _write_email_timing(dst_host, dst_user, dst_pass, execution_id, started_at, finished_at, error_msg=None):
    engine = None
    try:
        engine = sa.create_engine(
            f"postgresql://{quote_plus(dst_user)}:{quote_plus(dst_pass)}"
            f"@{dst_host}:5432/postgres",
            connect_args={'client_encoding': 'utf8'}
        )
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
                'execution_id':  execution_id,
                'pipeline_name': 'bronze_landing_zone',
                'activity_name': 'bronze_email_exporter',
                'started_at':    started_at,
                'finished_at':   finished_at,
                'duration_ms':   duration_ms,
                'status':        'ERROR' if error_msg else 'SUCCESS',
                'error_message': error_msg,
            })
    except Exception as te:
        print(f"WARN: No se pudo registrar tiempo email: {te}")
    finally:
        if engine:
            engine.dispose()


# ---------------------------------------------------------------------------
# Lectura de tiempos del pipeline desde la tabla
# ---------------------------------------------------------------------------

def _get_execution_times(dst_host, dst_user, dst_pass, execution_id):
    engine = None
    try:
        engine = sa.create_engine(
            f"postgresql://{quote_plus(dst_user)}:{quote_plus(dst_pass)}"
            f"@{dst_host}:5432/postgres",
            connect_args={'client_encoding': 'utf8'}
        )
        with engine.connect() as conn:
            rows = conn.execute(sa.text("""
                SELECT activity_name, started_at, finished_at, duration_ms, status
                FROM bronze._pipeline_execution_times
                WHERE execution_id = :eid
                ORDER BY started_at
            """), {'eid': execution_id}).fetchall()
        return [{'activity_name': r[0], 'started_at': r[1], 'finished_at': r[2],
                 'duration_ms': r[3], 'status': r[4]} for r in rows]
    except Exception as te:
        print(f"WARN: No se pudo leer tiempos de pipeline: {te}")
        return []
    finally:
        if engine:
            engine.dispose()

@data_exporter
def export_data(audit_summary, *args, **kwargs):
    _email_started_at = datetime.now()
    _email_error_msg  = None

    if not audit_summary:
        print("email_notifier: sin datos de audit_summary")
        return []
    if isinstance(audit_summary, dict):
        audit_summary = [audit_summary]

    smtp_host = _required_secret('smtp_host')
    smtp_port = _get_smtp_port()
    smtp_user = _required_secret('smtp-prod-user')
    smtp_pass = _required_secret('smtp-prod-password')

    destinatarios = _get_prod_recipients(kwargs)

    # Credenciales para auditoría y tiempos
    src_host = get_secret_value('BDWORKFLOW_HOST')
    src_user = get_secret_value('BDWORKFLOW_USER')
    src_pass = get_secret_value('BDWORKFLOW_PASSWORD')
    dst_host = get_secret_value('BDMANAGER_HOST')
    dst_user = get_secret_value('BDMANAGER_USER')
    dst_pass = get_secret_value('BDMANAGER_PASSWORD')

    # Extraction_id del run actual
    execution_id = next(
        (
            x.get('pipeline_execution_id')
            or x.get('execution_id')
            or x.get('extraction_id')
            for x in audit_summary
            if (x.get('pipeline_execution_id') or x.get('execution_id') or x.get('extraction_id'))
        ),
        'SIN_ID'
    )

    # --- KPIs exportación ---
    total_tables   = len(audit_summary)
    exportados_lst = [x for x in audit_summary if x.get('status') == 'EXPORTADO']
    auditables_lst = [x for x in audit_summary if str(x.get('status') or '').upper() in AUDITABLE_STATUSES]
    errores_lst    = [x for x in audit_summary if 'ERROR' in str(x.get('status', ''))]
    row_block_errors = [
        str(x.get('row_audit_block_error') or '').strip()
        for x in audit_summary
        if str(x.get('row_audit_block_status') or '').upper() == 'ERROR'
        and str(x.get('row_audit_block_error') or '').strip()
    ]
    schema_change_rows = [
        x for x in audit_summary
        if isinstance(x.get('schema_changes'), list) and len(x.get('schema_changes')) > 0
    ]
    row_block_error_msg = row_block_errors[0] if row_block_errors else None
    total_records  = sum(x.get('filas', 0) for x in exportados_lst)
    success_rate   = round(len(exportados_lst) / total_tables * 100, 2) if total_tables else 0
    runtime        = kwargs.get('runtime', 'N/A')
    run_id         = kwargs.get('run_id', execution_id)
    logs_url       = kwargs.get('logs_url', '#')
    environment    = kwargs.get('environment', 'Production')
    pipeline_name  = kwargs.get('pipeline_name', 'Pipeline DW Bronze')

    # --- Auditoría de filas (paralela) ---
    print("INFO: Iniciando auditoría de filas origen vs DW ...")
    row_audit_results = []
    try:
        if row_block_error_msg:
            print(f"WARN: Row audit transformer reporto error controlado. detalle={row_block_error_msg}")
            row_audit_results = [
                {
                    **x,
                    'source_count': x.get('source_count'),
                    'dw_count': x.get('dw_count'),
                    'difference': x.get('difference'),
                    'audit_status': 'ERROR_AUDITORIA',
                    'audit_reason': row_block_error_msg,
                }
                for x in auditables_lst
            ]
        else:
            row_audit_results = _build_row_audit_from_precomputed(audit_summary)
            if row_audit_results:
                print("INFO: Auditoría de filas tomada desde resultados precomputados.")
            else:
                row_audit_results = _build_row_audit(
                    audit_summary, src_host, src_user, src_pass, dst_host, dst_user, dst_pass
                )
    except Exception as ra_err:
        print(f"WARN: Error en auditoría de filas: {ra_err}")

    audit_ok       = [r for r in row_audit_results if r.get('audit_status') == 'OK']
    audit_disc     = [r for r in row_audit_results if r.get('audit_status') == 'DISCREPANCIA']
    audit_noaud    = [r for r in row_audit_results if r.get('audit_status') not in ('OK', 'DISCREPANCIA')]
    max_diff       = max((abs(r.get('difference') or 0) for r in audit_disc), default=0)
    data_integrity = round(len(audit_ok) / len(row_audit_results) * 100, 1) if row_audit_results else 100.0

    # --- Tiempos de pipeline desde BD ---
    timing_data = _get_execution_times(dst_host, dst_user, dst_pass, execution_id)

    # --- Excel adjuntos ---
    print("INFO: Generando adjuntos Excel ...")
    excel_export_bytes = _generate_excel_export_audit(audit_summary)
    excel_row_bytes    = _generate_excel_row_audit(row_audit_results)

    # --- Estado general del pipeline ---
    if success_rate == 100 and not audit_disc:
        pipeline_status = 'SUCCESS'
    elif errores_lst or audit_disc:
        pipeline_status = 'SUCCESS WITH WARNINGS' if exportados_lst else 'FAILED'
    else:
        pipeline_status = 'SUCCESS'

    # --- Datos para el template HTML ---
    top_errors = []
    for r in errores_lst[:5]:
        top_errors.append({
            'db':    r.get('db', ''),
            'tabla': r.get('target_table') or r.get('tabla', ''),
            'error': str(r.get('error', ''))[:120],
        })

    top_discrepancias = []
    for r in sorted(audit_disc, key=lambda x: abs(x.get('difference') or 0), reverse=True)[:5]:
        diff = r.get('difference', 0) or 0
        top_discrepancias.append({
            'tabla':        r.get('target_table') or r.get('tabla', ''),
            'source_count': r.get('source_count'),
            'dw_count':     r.get('dw_count'),
            'diff_label':   f"+{diff}" if diff > 0 else str(diff),
        })

    attention_items = []
    for r in errores_lst[:5]:
        attention_items.append(f"ERROR exportación: {r.get('db','')}.{r.get('target_table') or r.get('tabla','')}")
    if row_block_error_msg:
        attention_items.append(f"ERROR row_audit_transformer: {row_block_error_msg[:180]}")
    for r in audit_disc[:5]:
        diff = r.get('difference', 0) or 0
        attention_items.append(
            f"Discrepancia filas: {r.get('target_table') or r.get('tabla','')} "
            f"(origen={r.get('source_count')} DW={r.get('dw_count')} diff={'+' if diff>0 else ''}{diff})"
        )
    for r in schema_change_rows[:5]:
        table_name = r.get('target_table') or r.get('tabla', '')
        changes = r.get('schema_changes') or []
        cols = ', '.join([str(c.get('column')) for c in changes if c.get('column')])
        if cols:
            attention_items.append(f"Schema update: {table_name} (nuevas columnas: {cols})")
    attention_items = attention_items[:5]

    data = {
        'pipeline_name':      pipeline_name,
        'status':             pipeline_status,
        'environment':        environment,
        'run_id':             run_id,
        'runtime':            runtime,
        'datetime':           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'logs_url':           logs_url,
        # KPIs
        'total_tables':       total_tables,
        'tables_exported':    len(exportados_lst),
        'tables_failed':      len(errores_lst),
        'total_records':      total_records,
        'success_rate':       success_rate,
        'data_integrity_pct': data_integrity,
        # Audit export
        'export_warnings':    len(audit_summary) - len(exportados_lst) - len(errores_lst),
        'top_errors':         top_errors,
        # Audit filas
        'row_audit_ok_count':      len(audit_ok),
        'row_audit_disc_count':    len(audit_disc),
        'row_audit_noaud_count':   len(audit_noaud),
        'max_difference':          max_diff,
        'top_discrepancias':       top_discrepancias,
        'schema_change_count':     len(schema_change_rows),
        # Atención
        'attention_items':    attention_items,
        # Tiempos
        'timing_data':        timing_data,
    }

    html = render_email_html(data)

    subject = (
        f"[{environment}] {pipeline_name} — "
        f"{pipeline_status} | {len(exportados_lst)}/{total_tables} tablas"
    )

    msg = _build_message_with_attachments(
        subject, smtp_user, destinatarios, html,
        attachments=[
            ('export_audit.xlsx',        excel_export_bytes),
            ('row_validation_audit.xlsx', excel_row_bytes),
        ]
    )

    try:
        _send_email_with_retry(
            smtp_host, smtp_port, smtp_user, smtp_pass,
            destinatarios, msg, max_attempts=3, base_delay_seconds=2,
        )
    except Exception as main_error:
        _email_error_msg = str(main_error)[:200]
        print(f"ERROR: No se pudo enviar el correo principal de logs: {main_error}")
        try:
            _send_failure_notification(
                smtp_host, smtp_port, smtp_user, smtp_pass, destinatarios,
                pipeline_name, environment, run_id, main_error,
            )
        except Exception as failure_error:
            print(f"CRITICAL: Tampoco se pudo enviar notificacion de falla. error={failure_error}")
        raise
    finally:
        _email_finished_at = datetime.now()
        _write_email_timing(dst_host, dst_user, dst_pass, execution_id,
                            _email_started_at, _email_finished_at, _email_error_msg)

    print(
        f"Email enviado a {', '.join(destinatarios)}. "
        f"{len(exportados_lst)} exportados | {len(errores_lst)} errores | "
        f"filas OK={len(audit_ok)} DISC={len(audit_disc)}."
    )
    return audit_summary