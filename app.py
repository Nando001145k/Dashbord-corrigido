import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard PRF + INMET", layout="wide")
st.title("🛣️ Análise de Acidentes(PB)")
st.markdown("Dashboard interativo cruzando dados da **PRF** (2020 ~ 2025) **INMET** (via Acumulado Diário e Interpolação Espacial).")

# --- TRATAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    # 1. Carregar PRF
    df_prf = pd.read_csv('prf_pb_leve.csv')
    df_prf['data'] = pd.to_datetime(df_prf['data_inversa']).dt.date
    df_prf['hora_exata'] = pd.to_datetime(df_prf['horario'], format='%H:%M:%S', errors='coerce')
    df_prf['hora'] = df_prf['hora_exata'].dt.hour
    df_prf['municipio'] = df_prf['municipio'].str.upper().str.strip()
    
    # Arrumar caracteres estranhos
    df_prf['classificacao_acidente'] = df_prf['classificacao_acidente'].replace({
        'Com VÃ\xadtimas Feridas': 'Com Vítimas Feridas',
        'Sem VÃ\xadtimas': 'Sem Vítimas',
        'Com VÃ\xadtimas Fatais': 'Com Vítimas Fatais'
    })
    
    df_prf['tipo_acidente'] = df_prf['tipo_acidente'].str.replace('Ã£', 'ã').str.replace('Ã©', 'é').str.replace('Ã\xad', 'í')
    df_prf['causa_acidente'] = df_prf['causa_acidente'].str.replace('Ã£', 'ã').str.replace('Ã©', 'é').str.replace('Ã\xad', 'í').str.replace('Ã§', 'ç')
    
    df_prf['turno'] = pd.cut(
        df_prf['hora'], 
        bins=[-1, 5, 17, 24], 
        labels=['Noite/Madrugada', 'Dia', 'Noite/Madrugada'],
        ordered=False
    )
    
    # 2. Carregar INMET e Agregar por Dia (Soma da Chuva Diária)
    df_inmet = pd.read_csv('inmet_pb_leve.csv')
    df_inmet['data'] = pd.to_datetime(df_inmet['Data']).dt.date
    df_inmet['municipio'] = df_inmet['CIDADE'].str.upper().str.strip()
    coluna_chuva = df_inmet.columns[8] 
    df_inmet.rename(columns={coluna_chuva: 'chuva_mm'}, inplace=True)
    
    # Agregando o INMET por Dia e Estação (Soma total de chuva no dia)
    df_inmet_diario = df_inmet.groupby(['data', 'municipio'])['chuva_mm'].sum().reset_index()
    
    # 3. GEOMETRIA ESPACIAL (Fórmula de Haversine Completa para a Paraíba)
    estacoes_inmet = {
        'JOAO PESSOA': [-7.1150, -34.8631], 'CAMPINA GRANDE': [-7.2307, -35.8811],
        'PATOS': [-7.0244, -37.2801], 'AREIA': [-6.9667, -35.6983],
        'MONTEIRO': [-7.8894, -37.1203], 'CABACEIRAS': [-7.4892, -36.2869],
        'SAO GONCALO': [-6.75, -38.2166], 'ITAPORANGA': [-7.3039, -38.1517]
    }
    
    coordenadas = {
        'JOAO PESSOA': [-7.1150, -34.8631], 'CAMPINA GRANDE': [-7.2307, -35.8811],
        'SANTA RITA': [-7.1139, -34.9781], 'PATOS': [-7.0244, -37.2801],
        'BAYEUX': [-7.1250, -34.9322], 'CABEDELO': [-6.9811, -34.8339],
        'SOUSA': [-6.7583, -38.2269], 'CAJAZEIRAS': [-6.8894, -38.5606],
        'GUARABIRA': [-6.8522, -35.4897], 'AREIA': [-6.9667, -35.6983],
        'MONTEIRO': [-7.8894, -37.1203], 'POMBAL': [-6.7725, -37.7981],
        'MAMANGUAPE': [-6.8394, -35.1275], 'QUEIMADAS': [-7.3594, -35.9003],
        'ESPERANCA': [-7.0319, -35.8569], 'ITAPORANGA': [-7.3039, -38.1517],
        'CONDE': [-7.2597, -34.9075], 'ALHANDRA': [-7.4322, -34.9122],
        'PEDRAS DE FOGO': [-7.4019, -35.1164], 'SAPE': [-7.0969, -35.2269],
        'CATURITE': [-7.3850, -36.0500], 'BARRA DE SANTA ROSA': [-6.7269, -36.0758],
        'PICUI': [-6.5136, -36.3533], 'TAPEROA': [-7.2158, -36.8242],
        'JURIPIRANGA': [-7.2039, -35.1058], 'LUCENA': [-6.8908, -34.9136],
        'TEIXEIRA': [-7.2208, -37.2614], 'COREMAS': [-6.9678, -37.9536],
        'CATOLE DO ROCHA': [-6.3511, -37.7533], 'MONTE HOREBE': [-7.2519, -38.5447],
        'PRINCESA ISABEL': [-7.7375, -37.9958], 'SOLEDADE': [-7.0583, -36.3653],
        'POCINHOS': [-7.0678, -36.0694], 'SAO JOAO DO CARIRI': [-7.3889, -36.5292],
        'SERRA BRANCA': [-7.4811, -36.6558], 'INGA': [-7.3228, -35.6114],
        'GURINHEM': [-7.3458, -35.4536], 'REMIGIO': [-6.9972, -35.7958],
        'ALCANTIL': [-7.5186, -35.9406], 'SANTA TERESINHA': [-7.0508, -37.4589]
    }
    
    def calcular_distancia(lat1, lon1, lat2, lon2):
        if pd.isna(lat1) or pd.isna(lon1): return float('inf')
        R = 6371
        dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))

    def mapear_estacao(municipio):
        coords = coordenadas.get(municipio, [None, None])
        if coords[0] is None: 
            return 'CAMPINA GRANDE' 
        
        distancias = {estacao: calcular_distancia(coords[0], coords[1], ecoords[0], ecoords[1]) 
                      for estacao, ecoords in estacoes_inmet.items()}
        return min(distancias, key=distancias.get)

    df_prf['cidade_clima_referencia'] = df_prf['municipio'].apply(mapear_estacao)
    
    # 4. MERGE DIÁRIO (Cruzamento por Data e Estação Mais Próxima)
    df_merge = pd.merge(df_prf, df_inmet_diario[['data', 'municipio', 'chuva_mm']], 
                        left_on=['data', 'cidade_clima_referencia'],
                        right_on=['data', 'municipio'], 
                        how='left', suffixes=('', '_inmet'))
    
    df_merge['chuva_mm'] = df_merge['chuva_mm'].fillna(0)
    df_merge['condicao_clima'] = df_merge['chuva_mm'].apply(lambda x: 'Com Chuva' if x > 0 else 'Sem Chuva/Sem Registo')
    
    # Ajustes finais
    df_merge['ano'] = pd.to_datetime(df_merge['data_inversa']).dt.year
    df_merge['mes'] = pd.to_datetime(df_merge['data_inversa']).dt.month
    
    ordem_dias = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo']
    df_merge['dia_semana'] = df_merge['dia_semana'].str.replace('Ã§', 'ç').str.replace('sÃ¡bado', 'sábado')
    df_merge['dia_semana'] = pd.Categorical(df_merge['dia_semana'], categories=ordem_dias, ordered=True)
    df_merge['trecho'] = 'BR-' + df_merge['br'].astype(str) + ' | km ' + df_merge['km'].astype(str)
    
    return df_merge

df_completo = carregar_dados()

# --- MENU LATERAL (FILTROS) ---
st.sidebar.header("🎯 Controles da Análise")
ano_selecionado = st.sidebar.selectbox("Filtre pelo Ano:", ['Todos'] + sorted(list(df_completo['ano'].unique())))
br_selecionada = st.sidebar.selectbox("Rodovia (BR):", ['Todas'] + sorted(list(df_completo['br'].unique())))
gravidade_selecionada = st.sidebar.selectbox("Gravidade do Acidente:", ['Todas'] + list(df_completo['classificacao_acidente'].unique()))

# Aplicando os filtros globais
df_filtrado = df_completo.copy()
if ano_selecionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['ano'] == ano_selecionado]
if br_selecionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['br'] == br_selecionada]
if gravidade_selecionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['classificacao_acidente'] == gravidade_selecionada]

st.divider()

# ==========================================
# 📈 VISÃO GERAL
# ==========================================
st.markdown("## 📊 Visão Geral do Estado")

# Cards de Resumo (KPIs) - 5 colunas com Município Crítico e Ocorrências sob Chuva
col1, col2, col3, col4, col5 = st.columns(5)
total_acidentes = len(df_filtrado)
fatais = len(df_filtrado[df_filtrado['classificacao_acidente'].str.contains('Fatais', na=False)])
com_chuva = len(df_filtrado[df_filtrado['chuva_mm'] > 0])
taxa_letalidade = (fatais / total_acidentes * 100) if total_acidentes > 0 else 0

col1.metric("Total de Acidentes", total_acidentes)
col2.metric("Acidentes Fatais", fatais)
col3.metric("Taxa de Letalidade", f"{taxa_letalidade:.1f}%")
col4.metric("Ocorrências sob Chuva", com_chuva)

if not df_filtrado.empty:
    cidade_perigosa = df_filtrado['municipio'].value_counts().idxmax()
    col5.markdown("<p style='margin-bottom:-10px; font-size:14px; color:#555;'>Município Crítico</p>", unsafe_allow_html=True)
    col5.markdown(f"<h3 style='margin-top:0px; color:#31333F;'>{cidade_perigosa}</h3>", unsafe_allow_html=True)


# --- GRÁFICO DUPLO: ACIDENTES VS CHUVA ---
st.markdown("### 📅 Evolução Mensal: Volume de Acidentes vs Volume de Chuva")
sazonalidade = df_filtrado.groupby('mes').agg(
    Volume_Acidentes=('municipio', 'count'),
    Volume_Chuva=('chuva_mm', 'sum')
).reset_index()

fig_mes = go.Figure()
fig_mes.add_trace(go.Bar(x=sazonalidade['mes'], y=sazonalidade['Volume_Acidentes'], name="Nº de Acidentes", marker_color='#EF553B'))
fig_mes.add_trace(go.Scatter(x=sazonalidade['mes'], y=sazonalidade['Volume_Chuva'], name="Precipitação (mm)", mode='lines+markers', yaxis='y2', line=dict(color='#00CC96', width=3)))

fig_mes.update_layout(
    xaxis=dict(tickvals=list(range(1, 13)), title="Mês do Ano"),
    yaxis=dict(title="Quantidade de Acidentes", side='left'),
    yaxis2=dict(title="Volume de Chuva (mm)", side='right', overlaying='y', showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_mes, use_container_width=True)

st.divider()

# ==========================================
# 📊 VALIDAÇÃO DAS HIPÓTESES
# ==========================================
#st.markdown("## 🎯 Validação das Hipóteses de Pesquisa")

# --- HIPÓTESE 1: FATOR TURNO ---
st.markdown("### Severidade Noturna e Trechos Críticos")
col_h1a, col_h1b = st.columns(2)
with col_h1a:
    df_fatais_graves = df_filtrado[df_filtrado['classificacao_acidente'] != 'Sem Vítimas']
    turno_gravidade = df_fatais_graves.groupby('turno').size().reset_index(name='Quantidade')
    fig_turno = px.pie(turno_gravidade, values='Quantidade', names='turno', hole=0.5,
                       title="Acidentes com Vítimas por Turno", color='turno',
                       color_discrete_map={'Dia': '#FFC107', 'Noite/Madrugada': '#1A237E'})
    st.plotly_chart(fig_turno, use_container_width=True)

with col_h1b:
    trechos = df_filtrado['trecho'].value_counts().head(10).reset_index()
    trechos.columns = ['Trecho', 'Ocorrências']
    fig_trechos = px.bar(trechos, x='Ocorrências', y='Trecho', orientation='h',
                         title="Top 10 Trechos Mais Críticos", color='Ocorrências', color_continuous_scale='Reds')
    fig_trechos.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_trechos, use_container_width=True)


# --- HIPÓTESE 2: FATOR HUMANO ---
st.markdown("### O Fator mais comum de acidentes")
col_h2a, col_h2b = st.columns(2)

with col_h2a:
    causas = df_filtrado['causa_acidente'].value_counts().head(8).reset_index()
    causas.columns = ['Causa Principal', 'Quantidade']
    fig_causa = px.bar(causas, x='Causa Principal', y='Quantidade', 
                       title="Principais Causas Registradas", text_auto=True)
    st.plotly_chart(fig_causa, use_container_width=True)

with col_h2b:
    top_5_causas = df_filtrado['causa_acidente'].value_counts().head(5).index
    df_tree = df_filtrado[df_filtrado['causa_acidente'].isin(top_5_causas)]
    
    fig_tree = px.treemap(df_tree, path=['causa_acidente', 'tipo_acidente'], 
                          title="Dinâmica: Da Causa ao Tipo de Acidente (Clique para expandir)",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_tree, use_container_width=True)


# --- HIPÓTESE 3: SAZONALIDADE (COM HEATMAP) ---
st.markdown("### Sazonalidade (dias da Semana e Horários)")
col_h3a, col_h3b = st.columns(2)
with col_h3a:
    dia_semana_count = df_filtrado.groupby('dia_semana').size().reset_index(name='Total')
    fig_dias = px.line(dia_semana_count, x='dia_semana', y='Total', markers=True,
                       title="Volume de Acidentes por Dia da Semana")
    st.plotly_chart(fig_dias, use_container_width=True)

with col_h3b:
    fig_hora = px.histogram(df_filtrado, x='hora', nbins=24, 
                            title="Distribuição de Acidentes ao Longo do Dia (0h - 23h)",
                            labels={'hora': 'Hora do Dia (0h às 23h)'})
    fig_hora.update_layout(bargap=0.1)
    st.plotly_chart(fig_hora, use_container_width=True)

mapa_calor = df_filtrado.groupby(['dia_semana', 'hora']).size().reset_index(name='Total')
fig_heat = px.density_heatmap(
    mapa_calor, x='hora', y='dia_semana', z='Total',
    title="🚨 Mapa de Calor: Zonas Críticas na Semana (Dia vs Horário)",
    labels={'hora': 'Hora do Dia (0h-23h)', 'dia_semana': 'Dia da Semana', 'Total': 'Nº de Acidentes'},
    color_continuous_scale='Reds'
)
fig_heat.update_xaxes(dtick=1)
st.plotly_chart(fig_heat, use_container_width=True)


# --- HIPÓTESE 4: O RECORTE CLIMÁTICO (INMET) ---
st.markdown("### O Recorte Climático (Cruzamento INMET Diário via Interpolação)")
col_h4a, col_h4b = st.columns(2)
with col_h4a:
    impacto_chuva = df_filtrado.groupby(['condicao_clima', 'classificacao_acidente']).size().reset_index(name='Quantidade')
    
    fig_chuva = px.bar(impacto_chuva, x='condicao_clima', y='Quantidade', color='classificacao_acidente',
                       barmode='group', title="Gravidade dos Acidentes: Com Chuva vs Sem Chuva")
    st.plotly_chart(fig_chuva, use_container_width=True)

with col_h4b:
    tipos_chuva = df_filtrado[df_filtrado['condicao_clima'] == 'Com Chuva']
    if not tipos_chuva.empty:
        tipos_chuva_count = tipos_chuva['tipo_acidente'].value_counts().head(5).reset_index()
        tipos_chuva_count.columns = ['Tipo de Acidente (Com Chuva)', 'Ocorrências']
        
        fig_tipo_chuva = px.pie(tipos_chuva_count, values='Ocorrências', names='Tipo de Acidente (Com Chuva)',
                                title="Tipos de Acidentes mais comuns durante Chuva", hole=0.3)
        st.plotly_chart(fig_tipo_chuva, use_container_width=True)
    else:
        st.info("Não há dados de chuva registrados para os filtros selecionados.")