#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, base64, tempfile, warnings, chardet, logging, unicodedata
import pandas as pd, numpy as np
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash import callback_context

warnings.filterwarnings('ignore')

# =========================
# Config / logging
# =========================
CONFIG = {
    'SCALE_THRESHOLD': 0.90,
    'ENCODING_CONFIDENCE_MIN': 0.70,
    'REQUIRED_COLS': [
        'FECHA','MONTO (COP)','TIPO DE TRANSACCION','TIPO DE IDENTIFICACION',
        'ID DE CLIENTE','BANCO','BENEFICIARIO','ID DE TRANSACCION'
    ]
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Meses
_MESES_ES = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

# =========================
# Helpers
# =========================
def _strip_accents_lower(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def detectar_encoding(path, bytes_leer=100_000):
    try:
        with open(path, 'rb') as f: raw = f.read(bytes_leer)
        det = chardet.detect(raw) or {}
        enc, conf = det.get('encoding') or 'utf-8', det.get('confidence') or 0
        if conf < CONFIG['ENCODING_CONFIDENCE_MIN']:
            for enc_try in ['utf-8','latin-1','iso-8859-1','cp1252']:
                try:
                    with open(path, 'r', encoding=enc_try) as t: _ = t.read(1000)
                    return enc_try
                except:
                    pass
        return enc
    except:
        return 'utf-8'

def detectar_separador_csv(path, enc='utf-8', bytes_leer=5000):
    try:
        with open(path, 'r', encoding=enc, errors='ignore') as f:
            sample = f.read(bytes_leer)
        cand = {',': sample.count(','), ';': sample.count(';'), '\t': sample.count('\t'), '|': sample.count('|')}
        sep = max(cand.items(), key=lambda x: x[1])[0]
        return sep if cand[sep] > 0 else ','
    except:
        return ','

def optimizar_tipos_datos(df):
    for col in ['BANCO','TIPO DE TRANSACCION','TIPO DE IDENTIFICACION','TIPO_PERSONA','BENEFICIARIO']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    for col in df.select_dtypes(include=['float64']).columns:
        try:
            if df[col].notna().sum() and (df[col].dropna() % 1 == 0).all() and df[col].max() < 2_147_483_647:
                df[col] = df[col].astype('Int32')
        except:
            pass
    return df

def cargar_archivo_inteligente(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(file_path)
    elif ext == '.csv':
        enc = detectar_encoding(file_path)
        sep = detectar_separador_csv(file_path, enc)
        df = pd.read_csv(file_path, encoding=enc, sep=sep, low_memory=False)
    else:
        raise ValueError(f"Formato no soportado: {ext}")

    # Normalizaciones
    if 'TIPO DE TRANSACCION' in df.columns:
        df['TIPO DE TRANSACCION'] = df['TIPO DE TRANSACCION'].astype(str).map(_strip_accents_lower)

    if 'TIPO DE IDENTIFICACION' in df.columns:
        tip = df['TIPO DE IDENTIFICACION'].astype(str).str.strip().str.upper()
        df['TIPO_PERSONA'] = np.where(tip.str.startswith('N'), 'Jurídica', 'Natural')

    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')

    if 'MONTO (COP)' in df.columns:
        s = df['MONTO (COP)'].astype(str).str.strip().str.replace(r'[^0-9\-\., ]','', regex=True)
        def _to_float(x: str):
            x = x.replace(' ','')
            if x == '' or x is None: return np.nan
            c, p = x.rfind(','), x.rfind('.')
            if c > p:
                x = x.replace('.','').replace(',','.')
            elif p > c:
                x = x.replace(',','')
            else:
                x = x.replace(',','').replace('.','')
            try: return float(x)
            except: return np.nan
        df['MONTO (COP)'] = s.map(_to_float)

        serie = df['MONTO (COP)'].dropna().round()
        scale = 1.0
        if len(serie):
            if (serie % 100 == 0).mean() > CONFIG['SCALE_THRESHOLD']:
                scale = 0.01
            elif (serie % 10 == 0).mean() > CONFIG['SCALE_THRESHOLD']:
                scale = 0.10
        if scale != 1.0:
            df['MONTO (COP)'] *= scale
        df['MONTO (COP)'] = df['MONTO (COP)'].fillna(0.0)

    df = optimizar_tipos_datos(df)
    return df

def preparar_resultados_desde_df(df: pd.DataFrame, nombre_archivo: str):
    """Devuelve df_debito JSON (para store) + meta (diccionario con KPIs y listados)."""

    def fmt_cop(x):
        try: return f"$ {float(x):,.0f}".replace(',', '.')
        except: return str(x)

    def contar_tx(dframe):
        if 'ID DE TRANSACCION' in dframe.columns:
            return int(dframe['ID DE TRANSACCION'].nunique())
        return int(len(dframe))

    if df.empty:
        raise ValueError("El archivo no contiene datos utilizables.")

    # Filtrar a débito si existe la columna
    df_debito = df[df['TIPO DE TRANSACCION'] == 'debito'].copy() if 'TIPO DE TRANSACCION' in df.columns else df.copy()

    monto_total_archivo = df['MONTO (COP)'].sum() if 'MONTO (COP)' in df.columns else 0.0
    tx_total_archivo = contar_tx(df)

    monto_total_debito = df_debito['MONTO (COP)'].sum() if 'MONTO (COP)' in df_debito.columns else 0.0
    tx_total_debito = contar_tx(df_debito)

    monto_nat = monto_jur = 0.0; tx_nat = tx_jur = 0
    if not df_debito.empty and 'TIPO_PERSONA' in df_debito.columns:
        naturales = df_debito[df_debito['TIPO_PERSONA'] == 'Natural']
        juridicas = df_debito[df_debito['TIPO_PERSONA'] == 'Jurídica']
        monto_nat = naturales['MONTO (COP)'].sum(); monto_jur = juridicas['MONTO (COP)'].sum()
        tx_nat = contar_tx(naturales);            tx_jur = contar_tx(juridicas)

    if 'TIPO DE TRANSACCION' in df.columns:
        if 'ID DE TRANSACCION' in df.columns:
            tx_por_tipo = df.groupby('TIPO DE TRANSACCION')['ID DE TRANSACCION'].nunique().sort_values(ascending=False)
        else:
            tx_por_tipo = df['TIPO DE TRANSACCION'].value_counts()
    else:
        tx_por_tipo = pd.Series(dtype='int64')

    # Top 5 Bancos (monto) y (#TX)
    top_bancos_texto, top_bancos_transacciones = [], []
    if 'BANCO' in df_debito.columns:
        tb = (df_debito.groupby('BANCO')['MONTO (COP)'].sum().reset_index()
              .sort_values('MONTO (COP)', ascending=False).head(5))
        top_bancos_texto = [f"🏦 {r['BANCO']}: {fmt_cop(r['MONTO (COP)'])}" for _, r in tb.iterrows()]

        if 'ID DE TRANSACCION' in df_debito.columns:
            tx_por_banco = df_debito.groupby('BANCO')['ID DE TRANSACCION'].nunique()
        else:
            tx_por_banco = df_debito.groupby('BANCO').size()
        tb_tx = tx_por_banco.reset_index(name='TX').sort_values('TX', ascending=False).head(5)
        top_bancos_transacciones = [f"🔁 {r['BANCO']}: {int(r['TX']):,} TX".replace(',','.') for _, r in tb_tx.iterrows()]

    # Resumen temporal
    info_temporal = []
    if 'FECHA' in df_debito.columns:
        dfd = df_debito.dropna(subset=['FECHA'])
        if not dfd.empty:
            info_temporal.append(f"📅 Fecha más antigua: {dfd['FECHA'].min():%Y-%m-%d}")
            info_temporal.append(f"📅 Fecha más reciente: {dfd['FECHA'].max():%Y-%m-%d}")
            daily = dfd.groupby(dfd['FECHA'].dt.date)['MONTO (COP)'].sum().mean()
            info_temporal.append(f"💰 Promedio diario: {fmt_cop(daily)}")

    # DF para store (limpia bancos nulos/vacíos/"nan")
    cols_necesarias = [c for c in [
        'FECHA','BENEFICIARIO','BANCO','TIPO_PERSONA','MONTO (COP)','ID DE CLIENTE','ID DE TRANSACCION'
    ] if c in df_debito.columns]
    df_store = (df_debito[cols_necesarias]
                .dropna(subset=['FECHA','BANCO'])
                .loc[lambda d: d['BANCO'].astype(str).str.strip().ne('')]
                .loc[lambda d: ~d['BANCO'].astype(str).str.strip().str.lower().eq('nan')]
                .copy())

    meta = {
        'archivo': os.path.basename(nombre_archivo),
        'monto_total_archivo': fmt_cop(monto_total_archivo),
        'tx_total_archivo': f"{tx_total_archivo:,}".replace(',', '.'),
        'monto_total_debito': fmt_cop(monto_total_debito),
        'tx_total_debito': f"{tx_total_debito:,}".replace(',', '.'),
        'monto_nat': fmt_cop(monto_nat), 'tx_nat': f"{tx_nat:,}".replace(',', '.'),
        'monto_jur': fmt_cop(monto_jur), 'tx_jur': f"{tx_jur:,}".replace(',', '.'),
        'tx_por_tipo': tx_por_tipo.to_dict(),
        'top_bancos_texto': top_bancos_texto,
        'top_bancos_transacciones': top_bancos_transacciones,
        'info_temporal': info_temporal,
        'filas_originales': len(df),
        'filas_debito': len(df_debito),
    }
    return df_store.to_json(date_format='iso', orient='records'), meta

def procesar_upload(contents: str, filename: str):
    if not contents or not filename:
        return None, None, "❌ No se recibió archivo."
    tmp_path = None  # <- evita NameError en finally
    try:
        logging.info("🔔 on_upload: %s", filename)
        _, b64 = contents.split(',', 1)
        decoded = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or '.csv') as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        df = cargar_archivo_inteligente(tmp_path)
        if df is None or df.empty:
            return None, None, "⚠️ El archivo no contiene datos válidos."
        df_json, meta = preparar_resultados_desde_df(df, filename)
        return df_json, meta, f"✅ Archivo cargado: {filename} ({len(df):,} filas)".replace(',','.')
    except Exception as e:
        logging.exception("Error procesando archivo")
        return None, None, f"❌ Error procesando el archivo: {e}"
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass

# =========================
# App
# =========================
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css",
    ],
    title="Sistema de Análisis de Pagos",
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True
)

# Configuración del servidor
server = app.server

# ---- UI helpers ----
def crear_card(titulo, contenido, color="primary"):
    return dbc.Card(
        [dbc.CardHeader(html.H5(titulo, className="mb-0")), dbc.CardBody(contenido)],
        color=color, outline=True, className="mb-3"
    )

def crear_lista_items(items):
    if not items: return html.P("No hay datos disponibles", className="text-muted")
    return html.Ul([html.Li(i) for i in items], className="mb-0")

# ---- Vistas ----
# Función para encontrar los logos
def encontrar_logos():
    """Busca los logos específicos en la carpeta assets"""
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    if not os.path.exists(assets_dir):
        return None, None
    
    # Extensiones de imagen soportadas
    extensiones = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico']
    
    # Buscar los logos específicos
    logo_adamo_services = None
    logo_adamo_pay = None
    
    for archivo in os.listdir(assets_dir):
        if any(archivo.lower().endswith(ext) for ext in extensiones):
            if "Logo AdamoServices" in archivo:
                logo_adamo_services = f"/assets/{archivo}"
            elif "Logo AdamoPay" in archivo:
                logo_adamo_pay = f"/assets/{archivo}"
    
    return logo_adamo_services, logo_adamo_pay

def vista_gate(mensaje="Por favor, adjunta tu archivo para comenzar con el análisis."):
    # Buscar los logos
    logo_services, logo_pay = encontrar_logos()
    
    return [
        # Contenedor de header con logos y título
        dbc.Row([
            # Logo AdamoServices a la izquierda
            dbc.Col([
                html.Div([
                    html.A(
                        html.Img(
                            src=logo_services if logo_services else "https://via.placeholder.com/150x80?text=Logo AdamoServices",
                            alt="Logo Adamo Services",
                            className="logo-image logo-services",
                            style={"maxHeight": "60px", "width": "auto"}
                        ),
                        href="#",
                        className="logo-link"
                    )
                ], className="logo-container")
            ], width="auto", className="ms-3 d-flex align-items-center"),
            
            # Título centrado
            dbc.Col([
                html.H1("Adamo Services (AdamoPay_Análisis Transaccional)", 
                    className="text-center main-title mb-0 mt-5", 
                    style={
                        "color": "#111927 !important",  # Agregamos !important
                        "fontSize": "2.5rem",
                        "fontWeight": "600",
                        "letterSpacing": "0.5px",
                        "lineHeight": "1.0",
                        "paddingTop": "10px",
                        "paddingBottom": "10px"
                    })
            ], className="d-flex align-items-center justify-content-center"),
            
            # Logo AdamoPay a la derecha
            dbc.Col([
                html.Div([
                    html.A(
                        html.Img(
                            src=logo_pay if logo_pay else "https://via.placeholder.com/150x80?text=Logo AdamoPay",
                            alt="Logo AdamoPay",
                            className="logo-image logo-pay",
                            style={"maxHeight": "45px", "width": "auto"}
                        ),
                        href="#",
                        className="logo-link"
                    )
                ], className="logo-container")
            ], width="auto", className="me-3 d-flex align-items-center")
        ], className="mb-4 align-items-center"),
        
        # Contenido principal centrado con estilos específicos
        dbc.Row([
            dbc.Col([
                html.P(mensaje, className="text-center", 
                       style={"color": "#2C3E50 !important", "fontSize": "16px !important"}),
                html.Hr(),
                dbc.Card([
                    dbc.CardHeader(html.H5("📤 Cargar archivo", 
                                         style={"color": "#2C3E50 !important"})),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                dbc.Button(
                                    "Seleccionar archivo",
                                    id="upload-button",
                                    color="primary",
                                    className="me-2",
                                    style={
                                        "backgroundColor": "#5FE9D0",
                                        "borderColor": "#5FE9D0",
                                        "color": "#000000",
                                        "fontWeight": "500",
                                        "boxShadow": "none",
                                        "transition": "all 0.3s ease"
                                    },
                                ),
                                html.Span(
                                    "o arrastra y suelta aquí (.csv, .xlsx, .xls)",
                                    style={
                                        "color": "#2C3E50",
                                        "marginLeft": "10px",
                                        "verticalAlign": "middle"
                                    }
                                )
                            ], style={
                                "cursor": "pointer",
                                "padding": "15px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center"
                            }),
                            accept=".csv,.xlsx,.xls",
                            multiple=False,
                            style={"border":"1px dashed #bbb","borderRadius":"8px","padding":"30px","textAlign":"center"}
                        ),
                        html.Div(id='upload-status', className="mt-2", 
                               style={"color": "#2C3E50 !important"}),
                    ])
                ], className="mt-3")
            ], md=8, className="mx-auto")
        ])
    ]

def vista_dashboard(meta):
    header = dbc.Row([
        dbc.Col([
            html.H1(
                "📈 Reporte Análisis Transaccional AdamoPay",
                className="mb-2",
                style={'color': '#111927', 'fontSize': '2rem', 'fontWeight': '700'}
            ),
            html.P(
                f"Archivo procesado: {meta.get('archivo', '')}",
                style={'color': '#5a6268', 'fontSize': '1rem'}
            ),
        ]),
        dbc.Col(
            dbc.ButtonGroup([
                dbc.Button("Nuevo análisis", id="btn-new-analysis", color="secondary", outline=True, className="me-2"),
                dbc.Button("Recargar página", id="btn-reload", color="light"),
            ], className="float-end"), md=4, className="d-flex align-items-center justify-content-end"),
    ], className="mb-4 align-items-center")

    # KPIs
    totales = dbc.Row([
        dbc.Col(crear_card("📁 Información del Archivo", [
            html.P(f"📊 Filas originales: {meta['filas_originales']:,}".replace(",", ".")),
            html.P(f"📊 Filas de débito: {meta['filas_debito']:,}".replace(",", ".")),
        ], "info"), md=4),
        dbc.Col(crear_card("💰 Totales del Archivo", [
            html.H5(f"Monto total: {meta['monto_total_archivo']}", className="text-success"),
            html.H6(f"Transacciones: {meta['tx_total_archivo']}", className="text-info"),
        ]), md=4),
        dbc.Col(crear_card("💳 Totales Débito", [
            html.H5(f"Monto débito: {meta['monto_total_debito']}", className="text-success"),
            html.H6(f"Transacciones débito: {meta['tx_total_debito']}", className="text-info"),
        ]), md=4),
    ], className="mb-4")

    # Segmentos
    segmentos = dbc.Row([
        dbc.Col(crear_card("👥 Personas Naturales", [
            html.H5(f"Monto: {meta['monto_nat']}", className="text-success"),
            html.P(f"Transacciones: {meta['tx_nat']}", className="text-info"),
        ], "success"), md=6),
        dbc.Col(crear_card("🏢 Personas Jurídicas", [
            html.H5(f"Monto: {meta['monto_jur']}", className="text-success"),
            html.P(f"Transacciones: {meta['tx_jur']}", className="text-info"),
        ], "success"), md=6),
    ], className="mb-4")

    # Gráficas
    charts = dbc.Row([
        dbc.Col(crear_card("📊 Análisis transaccional | 🗓️ Comportamiento Mensual", [
            dbc.Row([
                dbc.Col([
                    html.Label("Seleccionar Mes", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='mes-select', 
                        options=[], 
                        value=None, 
                        clearable=False,
                        persistence=True, 
                        persistence_type='session',
                        style={
                            'fontSize': '14px',
                            'borderRadius': '8px',
                            'border': '1px solid #dee2e6'
                        }
                    )
                ], md=4),
                dbc.Col([
                    html.Div([
                        html.I(className="bi bi-info-circle me-2"),
                        html.Span("Gráfico interactivo: Se identifica comportamiento mensual y el día de mayor monto de cada semana", 
                                className="small text-muted")
                    ], className="mt-4")
                ], md=8)
            ], className="mb-3"),
            dcc.Graph(
                id='g-tiempo',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'analisis_temporal',
                        'height': 500,
                        'width': 1200,
                        'scale': 1
                    }
                },
                style={'height': '450px', 'borderRadius': '8px', 'border': '1px solid #e9ecef'}
            )
        ], color="primary"), md=12),

        dbc.Col(crear_card("🏛️ Distribución por Entidades Bancarias", [
            html.Div([
                html.P("Análisis comparativo de montos procesados por banco en el período seleccionado", 
                       className="text-muted mb-3 small"),
                dcc.Graph(
                    id='g-bancos',
                    config={
                        'displayModeBar': True,
                        'displaylogo': True,
                        'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'analisis_bancos',
                            'height': 600,
                            'width': 1500,
                            'scale': 1
                        }
                    },
                    style={'borderRadius': '8px', 'border': '1px solid #e9ecef'}
                )
            ])
        ], color="info"), md=12),

        dbc.Col(crear_card("🏆 Ranking de Entidades - Monto Total", [
            html.Div([
                html.H6("Top 5 Bancos por Volumen Monetario", className="text-primary mb-3"),
                crear_lista_items(meta.get('top_bancos_texto', [])),
                html.Hr(className="my-3"),
                html.Small("💡 Datos basados en transacciones de débito procesadas", 
                          className="text-muted")
            ])
        ], color="success"), md=6),
        
        dbc.Col(crear_card("📊 Ranking de Entidades - Volumen Transaccional", [
            html.Div([
                html.H6("Top 5 Bancos por Número de Transacciones", className="text-info mb-3"),
                crear_lista_items(meta.get('top_bancos_transacciones', [])),
                html.Hr(className="my-3"),
                html.Small("🔢 Conteo de transacciones únicas por entidad bancaria", 
                          className="text-muted")
            ])
        ], color="warning"), md=6),
    ])

    # Ranking
    ranking = dbc.Row([
        dbc.Col(crear_card("🏆 Ranking de Clientes", [
            dbc.Row([
                dbc.Col([
                    html.Label("Segmento"),
                    dcc.RadioItems(
                        id='rank-segmento',
                        options=[{'label':'Naturales','value':'Natural'},
                                 {'label':'Jurídicas','value':'Jurídica'}],
                        value='Natural', inline=True
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Métrica"),
                    dcc.RadioItems(
                        id='rank-metrica',
                        options=[{'label':'Monto (COP)','value':'MONTO'},
                                 {'label':'# Transacciones','value':'TX'}],
                        value='MONTO', inline=True
                    )
                ], md=4),
                dbc.Col([html.Label("Top N"),
                         dcc.Input(id='rank-topn', type='number', value=10, min=3, max=50, step=1, style={'width':'100%'})], md=2),
                dbc.Col([html.Label("Descargar"),
                         html.Div([dbc.Button("CSV", id='btn-dl-rank', color='primary', className='me-2'),
                                   dcc.Download(id='dl-rank')])], md=2),
            ], className='mb-3'),

            dash_table.DataTable(
                id='tabla-ranking',
                columns=[
                    {"name": "Beneficiario", "id": "BENEFICIARIO"},
                    {"name": "Monto (COP)", "id": "MONTO_STR"},
                    {"name": "# Transacciones", "id": "TX"},
                    {"name": "ID de Cliente", "id": "ID DE CLIENTE"},
                    {"name": "% Participación", "id": "PCT_STR"}
                ],
                data=[],
                page_size=12,
                style_table={'overflowX': 'auto'},
                style_cell_conditional=[],
                style_data={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'},
                page_action='native',
                filter_action='none',
                sort_action='none'
            ),
            html.Div(className='mt-3'),
            dcc.Graph(id='graf-ranking')
        ]), md=12)
    ])

    # Resumen
    resumen = dbc.Row([
        dbc.Col(crear_card("📅 Análisis Temporal (resumen)", [crear_lista_items(meta.get('info_temporal', []))]), md=12),
    ])

    footer = html.Footer([
        html.Hr(),
        html.P("✅ Análisis completado", className="text-center text-muted"),
        html.P(f"Procesado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", className="text-center text-muted small")
    ])

    return [header, totales, segmentos, charts, ranking, resumen, footer]

# ---- Layout base ----
default_figure = px.scatter(title='Sin datos disponibles').update_layout(
    height=300,
    showlegend=False,
    xaxis={'visible': False},
    yaxis={'visible': False},
    annotations=[{
        'text': 'No hay datos para mostrar',
        'xref': 'paper',
        'yref': 'paper',
        'showarrow': False,
        'font': {'size': 40}
    }]
)

app.layout = dbc.Container([
    # Stores para datos
    dcc.Store(id='data-store', data=None, storage_type='memory'),
    dcc.Store(id='meta-store', data=None, storage_type='memory'),
    
    # Contenedor principal
    html.Div(
        id='app-body',
        children=vista_gate("Por favor, adjunta tu archivo para comenzar con el análisis"),
        style={'color': '#2C3E50','fontSize': '16px'}
    ),
    
    # Componente para refresh
    html.Div(id='dummy-refresh'),
], fluid=True)

# ---- Validation layout ----

app.validation_layout = html.Div([
    # Layout base
    app.layout,
    # Componentes adicionales que pueden aparecer dinámicamente
    html.Div([
        dcc.Upload(id='upload-data-validation'),
        html.Div(id='upload-status-validation'),
        dcc.Dropdown(id='mes-select-validation'),
        dcc.Graph(id='g-tiempo-validation'),
        dcc.Graph(id='g-bancos-validation'),
        dash_table.DataTable(
            id='tabla-ranking-validation',
            columns=[],
            data=[]
        ),
        dcc.Graph(id='graf-ranking-validation'),
        dcc.Download(id='dl-rank-validation'),
        dbc.Button(id='btn-new-analysis-validation'),
        dbc.Button(id='btn-reload-validation'),
        dbc.Button(id='btn-dl-rank-validation'),
        dcc.RadioItems(id='rank-segmento-validation'),
        dcc.RadioItems(id='rank-metrica-validation'),
        dcc.Input(id='rank-topn-validation')
    ])
])

# =========================
# Router (Gate ↔ Dashboard)
# =========================
@app.callback(
    Output('app-body', 'children'),
    Input('meta-store', 'data'),
    prevent_initial_call=True
)
def render_body(meta):
    """Si no hay meta => Gate (subida). Si hay meta => Dashboard."""
    try:
        if meta is None:
            return vista_gate("Por favor, adjunta tu archivo para comenzar con el análisis")
        return vista_dashboard(meta)
    except Exception as e:
        logging.exception("Error en render_body")
        return vista_gate(f"Error: {str(e)}")

# =========================
# ✅ CALLBACK CORREGIDO - Combinado para Upload y Nuevo análisis
# =========================
@app.callback(
    Output('data-store', 'data'),
    Output('meta-store', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    Input('btn-new-analysis', 'n_clicks'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def manejar_datos_y_reset(contents, btn_clicks, filename):
    """Maneja tanto la carga de archivos como el reset del análisis"""
    ctx = callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Si el trigger es el botón "Nuevo análisis"
    if trigger_id == 'btn-new-analysis' and btn_clicks:
        logging.info("🔄 Reiniciando análisis - volviendo al gate")
        return None, None, "Por favor, adjunta tu archivo para comenzar con el análisis"
    
    # Si el trigger es la carga de archivo
    elif trigger_id == 'upload-data' and contents and filename:
        logging.info(f"📤 Procesando archivo cargado: {filename}")
        df_json, meta, msg = procesar_upload(contents, filename)
        return df_json, meta, msg
    
    # Caso por defecto
    raise PreventUpdate

# =========================
# Recargar página (client-side)
# =========================
app.clientside_callback(
    """
    function(n){
        if(n){ window.location.reload(); }
        return '';
    }
    """,
    Output('dummy-refresh', 'children'),
    Input('btn-reload', 'n_clicks'),
    prevent_initial_call=True
)

# =========================
# ✅ CALLBACK PROTEGIDO - Poblar selector de mes
# =========================
@app.callback(
    Output('mes-select', 'options'),
    Output('mes-select', 'value'),
    Input('data-store', 'data'),
    prevent_initial_call=True
)
def poblar_selector_mes(json_data):
    if not json_data:
        raise PreventUpdate
    
    try:
        df = pd.read_json(json_data)
        if df.empty or 'FECHA' not in df.columns:
            raise PreventUpdate

        df['FECHA'] = pd.to_datetime(df['FECHA'])
        meses_ord = (df.dropna(subset=['FECHA'])
                       .assign(_MES_KEY=df['FECHA'].dt.to_period('M').dt.to_timestamp())
                       .sort_values('_MES_KEY')['_MES_KEY'].drop_duplicates().tolist())

        options = [{'label': f"{_MESES_ES[m.month-1]} {m.year}", 'value': f"{m.year:04d}-{m.month:02d}"} for m in meses_ord]
        default_val = options[-1]['value'] if options else None
        return options, default_val
    except:
        raise PreventUpdate

# =========================
# ✅ CALLBACK PROTEGIDO - Gráficas principales
# =========================
@app.callback(
    Output('g-tiempo','figure'),
    Output('g-bancos','figure'),
    Input('data-store','data'),
    Input('mes-select','value'),
    prevent_initial_call=True
)
def actualizar_graficas(json_data, mes_sel):
    if not json_data:
        raise PreventUpdate
    
    try:
        df = pd.read_json(json_data)
        if df.empty or 'FECHA' not in df.columns:
            raise PreventUpdate

        df['FECHA'] = pd.to_datetime(df['FECHA'])
        if not mes_sel:
            ult = (df['FECHA'].dt.to_period('M').dt.to_timestamp()).max()
            mes_sel = f"{ult.year:04d}-{ult.month:02d}"

        y, m = map(int, mes_sel.split('-'))
        dfd = df[(df['FECHA'].dt.year == y) & (df['FECHA'].dt.month == m)].copy()

        # ========= Gráfico 1: Monto diario (Débito) =========
        if dfd.empty:
            fig_tiempo = px.scatter(title='Sin datos para el mes seleccionado')
        else:
            dfd['DIA'] = dfd['FECHA'].dt.floor('D')
            month_start = dfd['FECHA'].dt.to_period('M').dt.start_time
            dfd['SEMANA'] = ((dfd['FECHA'] - month_start).dt.days // 7) + 1

            serie = (dfd.groupby(['DIA','SEMANA'])['MONTO (COP)']
                     .sum().reset_index().rename(columns={'MONTO (COP)':'MONTO'}))

            idx_max = serie.groupby('SEMANA')['MONTO'].idxmax()
            picos = serie.loc[idx_max].sort_values('DIA')

            # Paleta de colores profesional (tonos de azul y gris)
            colores_semana = {
                1: "#B8B8B8",  
                2: "#90caf9",  
                3: "#1C6CF8",  
                4: "#6C737F",
                5: "#80CCEB"
            }
            day_ms = 24*60*60*1000

            fig_tiempo = go.Figure()
            # Variables para control de ancho y alineación
            bar_width = 0.85 * day_ms
            
            # Primero agregamos las barras de fondo (los montos por día)
            for sem, dfw in serie.groupby('SEMANA'):
                fig_tiempo.add_trace(go.Bar(
                    x=dfw['DIA'], y=dfw['MONTO'], name=f"Semana {int(sem)}",
                    marker_color=colores_semana.get(int(sem), "#e02929"),
                    marker_line_color="rgba(0,0,0,0)",
                    width=[bar_width]*len(dfw),
                    hovertemplate="Fecha %{x}<br>Monto $ %{y:,.0f}<extra></extra>",
                    base=0
                ))

            # Para los picos, en lugar de un marcador, cambiaremos el color de la barra
            # y agregaremos una anotación más clara.
            pico_dias = picos['DIA'].tolist()
            
            # Iteramos sobre los picos para agregar anotaciones y resaltar las barras
            for _, pico in picos.iterrows():
                # Cambiar el color de la barra del día pico
                # Buscamos la traza de la semana correspondiente
                for trace in fig_tiempo.data:
                    if trace.name == f"Semana {int(pico['SEMANA'])}":
                        # Creamos una lista de colores para esta traza
                        colores_barra = [trace.marker.color] * len(trace.x)
                        try:
                            # Encontramos el índice del día pico en la traza
                            idx = list(trace.x).index(pico['DIA'])
                            # Cambiamos el color de esa barra específica
                            # No se cambia el color, se mantiene el de la semana.
                            # El resaltado se logra con el borde y la anotación.
                            pass # Mantenemos el color original de la semana
                            trace.marker.color = colores_barra
                            # Agregamos un borde para que resalte más
                            line_colors = ['rgba(0,0,0,0)'] * len(trace.x)
                            line_colors[idx] = 'black'
                            trace.marker.line.color = line_colors
                            trace.marker.line.width = 1.5
                        except ValueError:
                            # El día pico no está en esta traza (no debería pasar)
                            pass

                # Agregar una anotación en lugar de un marcador de texto
                fig_tiempo.add_annotation(
                    x=pico['DIA'],
                    y=pico['MONTO'],
                    text=f"💲Día Pico:<br><b>${pico['MONTO']:,.0f}</b>".replace(',', '.'),
                    width=100,
                    height=30,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#343a40",
                    font=dict(size=12, color="black"),
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2,  # Ajusta el padding dentro de la caja de anotación
                    ax=0,  # Centra la anotación horizontalmente
                    ay=-40, # Ajusta la posición vertical para que esté sobre la barra
                    xanchor="center",
                    align="center"
                )
            
            # Añadir una leyenda falsa para explicar el resaltado del día pico.
            # La posición de la leyenda se ajusta en `update_layout` más adelante.
            fig_tiempo.add_trace(go.Bar(
                x=[None], y=[None], name="Día Pico Semanal",
                marker=dict(color="rgba(255, 255, 255, 0.85)", line=dict(color='black', width=1.5))
            ))

            dias_dt = pd.to_datetime(serie['DIA']); dmin, dmax = dias_dt.min(), dias_dt.max()
            fig_tiempo.update_xaxes(
                range=[dmin - pd.Timedelta(days=0.8), dmax + pd.Timedelta(days=1.5)],
                tickformat="%d-%b", tickangle=-35, dtick="D1",
                tickmode="linear", showgrid=True, showline=True, linewidth=0.7, linecolor='black', automargin=True,
                gridcolor='rgba(0,0,0,0.1)', title='Día (mes seleccionado)'
            )
            ymax = float(max(serie['MONTO'].max(), picos['MONTO'].max()))
            fig_tiempo.update_yaxes(
                range=[0, ymax * 1.25] if ymax > 0 else None,
                tickformat=',.0f', tickprefix='$ ', title='Monto (COP)',
                gridcolor='rgba(0,0,0,0.1)', showgrid=True, automargin=True,
                showline=True, side="left", linewidth=0.7, linecolor='black'
            )
            fig_tiempo.update_layout(
                title={'text': f"Monto diario (COP) — {_MESES_ES[m-1]} {y}", 'x': 0.02, 'y': 0.98, 'xanchor': 'left', 'yanchor': 'top'},
                font=dict(family="Arial, Segoe UI, Roboto", size=12, color="#222"),
                barmode='stack',  # Asegura que las barras se centren
                bargap=0.2,       # Espacio entre barras de diferentes días
                plot_bgcolor="#f7fbf6",
                paper_bgcolor="white",
                uirevision=f"mes-{mes_sel}",
                legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="left", x=-0.08,
                            bgcolor="rgba(255,255,255,0.6)", bordercolor="lightgray", borderwidth=1)
            )

        # ========= Gráfico 2: Bancos por monto (todos los bancos) =========
        if dfd.empty or 'BANCO' not in dfd.columns:
            fig_bancos = px.scatter(title='Sin datos para Bancos')
        else:
            dfb = dfd.copy()
            mask = dfb['BANCO'].notna()
            mask &= dfb['BANCO'].astype(str).str.strip().ne('')
            mask &= ~dfb['BANCO'].astype(str).str.strip().str.lower().eq('nan')
            dfb = dfb.loc[mask]
            if dfb.empty:
                fig_bancos = px.scatter(title='Sin datos para bancos válidos')
            else:
                bancos_sum = (dfb.groupby('BANCO')['MONTO (COP)']
                                .sum().reset_index())
                orden = bancos_sum.sort_values('MONTO (COP)', ascending=True)
                total = float(orden['MONTO (COP)'].sum())
                orden['PCT'] = np.where(total > 0, orden['MONTO (COP)'] / total, 0.0)

                # Crear una escala de colores basada en los montos
                n_bancos = len(orden)
                # Crear una gradiente de colores más atractiva
                colors = []
                for i in range(n_bancos):
                    if i == n_bancos - 1:  # El banco con mayor monto
                        colors.append("#5FE9D0")  # Verde-azul distintivo
                    elif i == n_bancos - 2:  # Segundo banco
                        colors.append("#90CAF9")  # Azul medio
                    elif i == n_bancos - 3:  # Tercer banco
                        colors.append("#1C6CF8")  # Azul vibrante
                    else:
                        colors.append("#B8B8B8")  # Gris para el resto

                # Crear la gráfica con escala logarítmica
                fig_bancos = go.Figure(go.Bar(
                    x=orden['MONTO (COP)'],
                    y=orden['BANCO'].astype(str),
                    orientation='h',
                    marker_color=colors,
                    marker_line_color="black",
                    marker_line_width=0.8,
                    text=[f"${v:,.0f}".replace(',', '.') for v in orden['MONTO (COP)']],
                    textposition="outside",
                    textfont=dict(
                        size=11,
                        color="#111927",
                        family="Arial, Segoe UI, Roboto"
                    ),
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>" +
                        "<b>Monto:</b> $ %{x:,.0f}<br>" +
                        "<b>Participación:</b> %{customdata:.1f}%<extra></extra>"
                    ),
                    customdata=orden['PCT']*100,
                    width=0.7,  # Controla el ancho de las barras
                ))

                n_bancos = len(orden)
                left_margin = max(180, int(7.0 * orden['BANCO'].astype(str).str.len().max()))
                xmax = float(orden['MONTO (COP)'].max()) if n_bancos else 0.0

                fig_bancos.update_layout(
                    title={
                        'text': f"Distribución de Montos por Banco — {_MESES_ES[m-1]} {y}",
                        'x': 0.02,
                        'y': 0.97,
                        'xanchor': 'left',
                        'yanchor': 'top',
                        'font': dict(
                            size=15,
                            color="#111927",
                            family="Arial, Segoe UI, Roboto"
                        )
                    },
                    height=max(450, min(1400, 180 + 35 * n_bancos)),
                    margin=dict(l=left_margin, r=120, t=100, b=50),
                    plot_bgcolor="#f8fafc",
                    paper_bgcolor="white",
                    bargap=0.35,
                    font=dict(
                        family="Arial, Segoe UI, Roboto",
                        size=16,
                        color="#111927"
                    ),
                    showlegend=False,
                    uirevision=f"mes-{mes_sel}-bancos"
                )
                # Configuración del eje X (montos) con escala logarítmica
                fig_bancos.update_xaxes(
                    # Se cambia a escala lineal para mejor proporción visual
                    type="linear",
                    title={
                        'text': "Monto Total (COP)",
                        'font': dict(
                            size=13,
                            color="#111927",
                            family="Arial, Segoe UI, Roboto"
                        ),
                        'standoff': 15
                    },
                    # Ajustar el rango para dar espacio a las etiquetas de texto
                    range=[0, xmax * 1.35] if xmax > 0 else None,
                    tickformat=",",
                    tickprefix="$ ",
                    showline=True,
                    linewidth=1,
                    linecolor="#2c3e50",
                    gridcolor="rgba(0,0,0,0.05)",
                    automargin=True,
                    tickfont=dict(
                        size=11,
                        family="Arial, Segoe UI, Roboto"
                    )
                )
                
                # Configuración del eje Y (bancos)
                fig_bancos.update_yaxes(
                    title={
                        'text': "Entidad Bancaria",
                        'font': dict(
                            size=13,
                            color="#111927",
                            family="Arial, Segoe UI, Roboto"
                        )
                    },
                    categoryorder="array",
                    categoryarray=orden['BANCO'].tolist(),
                    showline=True,
                    linewidth=1,
                    linecolor="#111927",
                    automargin=True,
                    tickfont=dict(
                        size=12,
                        family="Arial, Segoe UI, Roboto",
                        color="#111927"
                    )
                )

        return fig_tiempo, fig_bancos
    except:
        raise PreventUpdate

# =========================
# ✅ RANKING PROTEGIDO
# =========================
def _fmt_cop_str(x: float) -> str:
    try: return f"$ {float(x):,.0f}".replace(',', '.')
    except: return str(x)

def _construir_ranking(df: pd.DataFrame, segmento: str, metrica: str, topn: int) -> pd.DataFrame:
    cols_out = ['BENEFICIARIO','ID DE CLIENTE','MONTO','MONTO_STR','TX']
    if df.empty: return pd.DataFrame(columns=cols_out)

    base = df[df['TIPO_PERSONA'] == segmento].copy()
    if base.empty: return pd.DataFrame(columns=cols_out)

    keys = []
    if 'ID DE CLIENTE' in base.columns: keys.append('ID DE CLIENTE')
    if 'BENEFICIARIO' in base.columns: keys.append('BENEFICIARIO')
    if not keys:
        keys = ['BANCO'] if 'BANCO' in base.columns else []
        if not keys: return pd.DataFrame(columns=cols_out)

    if 'ID DE TRANSACCION' in base.columns:
        agg_tx = base.groupby(keys)['ID DE TRANSACCION'].nunique()
    else:
        agg_tx = base.groupby(keys).size()
    agg_monto = base.groupby(keys)['MONTO (COP)'].sum()

    tabla = pd.concat([agg_monto.rename('MONTO'), agg_tx.rename('TX')], axis=1).reset_index()
    if metrica == 'TX':
        tabla = tabla.sort_values(['TX','MONTO'], ascending=[False, False])
    else:
        tabla = tabla.sort_values(['MONTO','TX'], ascending=[False, False])

    tabla['MONTO_STR'] = tabla['MONTO'].map(_fmt_cop_str)
    for c in cols_out:
        if c not in tabla.columns: tabla[c] = ''
    return tabla.head(int(topn))[cols_out]

@app.callback(
    Output('tabla-ranking','data'),
    Output('graf-ranking','figure'),
    Input('data-store','data'),
    Input('rank-segmento','value'),
    Input('rank-metrica','value'),
    Input('rank-topn','value'),
    prevent_initial_call=True
)
def actualizar_ranking(json_data, segmento, metrica, topn):
    # Protección contra valores undefined o inválidos
    if not json_data or segmento is None or metrica is None or topn is None:
        raise PreventUpdate

    try:
        df = pd.read_json(json_data)

        topn = int(topn or 10)
        tabla = _construir_ranking(df, segmento or 'Natural', metrica or 'MONTO', topn)
        if tabla.empty:
            return [], px.scatter(title='Sin datos para el ranking')

        df_seg = df[df['TIPO_PERSONA'] == (segmento or 'Natural')].copy()
        if (metrica or 'MONTO') == 'TX':
            total_metric = df_seg['ID DE TRANSACCION'].nunique() if 'ID DE TRANSACCION' in df_seg.columns else len(df_seg)
            tabla['PCT'] = np.where(total_metric > 0, tabla['TX'] / total_metric, 0.0)
            y_col, y_title, fig_title = 'TX', '# Transacciones', f"Top {len(tabla)} {segmento} por # Transacciones"
        else:
            total_metric = df_seg['MONTO (COP)'].sum()
            tabla['PCT'] = np.where(total_metric > 0, tabla['MONTO'] / total_metric, 0.0)
            y_col, y_title, fig_title = 'MONTO', 'Monto (COP)', f"Top {len(tabla)} {segmento} por Monto (COP)"

        tabla['PCT_STR'] = (tabla['PCT'] * 100).map(lambda v: f"{v:,.1f} %".replace(",", "."))
        if 'ID DE CLIENTE' not in tabla.columns:
            tabla['ID DE CLIENTE'] = ''

        fig = px.bar(tabla, x='BENEFICIARIO', y=y_col, title=fig_title, text='PCT_STR')
        fig.update_layout(xaxis_title='Beneficiario', yaxis_title=y_title)

        return tabla[['BENEFICIARIO','MONTO_STR','TX','PCT_STR','ID DE CLIENTE']].to_dict(orient='records'), fig
    except:
        raise PreventUpdate

# =========================
# ✅ DESCARGA PROTEGIDA
# =========================
@app.callback(
    Output('dl-rank', 'data'),
    Input('btn-dl-rank', 'n_clicks'),
    State('tabla-ranking', 'data'),
    prevent_initial_call=True
)
def descargar_ranking(n_clicks, data):
    if not n_clicks or not data:
        raise PreventUpdate
    try:
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, filename=f"ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    except:
        raise PreventUpdate

# =========================
# Main
# =========================

if __name__ == '__main__':
    # Configuración para Railway
    port = int(os.environ.get("PORT", 8050))
    debug_mode = os.environ.get("DASH_DEBUG", "False").lower() == "true"
    
    print("="*70)
    print("🚀 INICIANDO SISTEMA DE ANÁLISIS DE PAGOS WEB")
    print(f"Puerto: {port}")
    print(f"Debug: {debug_mode}")
    print("="*70)
    
    # En producción, Gunicorn manejará el servidor
    # Este bloque solo se ejecuta en desarrollo local
    app.run_server(
        debug=debug_mode, 
        host='0.0.0.0', 
        port=port,
        dev_tools_ui=debug_mode,
        dev_tools_props_check=debug_mode
    )