import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ---------------------------------------------------------------------------
# 1. Leitura e preparação dos dados
# ---------------------------------------------------------------------------
df = pd.read_csv('ecommerce_estatistica.csv')

mapa_genero = {
    'Masculino': 'Masculino',
    'Feminino': 'Feminino',
    'Sem gênero': 'Sem gênero / Unissex',
    'Unissex': 'Sem gênero / Unissex',
    'Meninos': 'Infantil',
    'Meninas': 'Infantil',
    'Sem gênero infantil': 'Infantil',
    'Bebês': 'Infantil',
}
df['Genero_Grupo'] = df['Gênero'].map(mapa_genero).fillna('Sem gênero / Unissex')
df['Marca_Limpa'] = df['Marca'].str.strip().str.lower()

COLUNAS_NUM = ['Nota', 'N_Avaliações', 'Desconto', 'Preço',
               'Qtd_Vendidos_Cod', 'Marca_Freq', 'Material_Freq']

GENEROS = sorted(df['Genero_Grupo'].unique())
PRECO_MIN = int(np.floor(df['Preço'].min()))
PRECO_MAX = int(np.ceil(df['Preço'].max()))


def figura_vazia(titulo):
    """Figura exibida quando os filtros não retornam nenhum produto."""
    fig = go.Figure()
    fig.update_layout(
        title=titulo,
        annotations=[dict(text='Nenhum produto com os filtros selecionados',
                          showarrow=False, font=dict(size=14))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def curva_densidade(valores, pontos=200):
    """Estimativa de densidade (KDE gaussiana) usando apenas numpy."""
    valores = np.asarray(valores, dtype=float)
    if len(valores) < 2 or valores.std() == 0:
        return None, None
    h = 1.06 * valores.std() * len(valores) ** (-1 / 5)  # regra de Silverman
    x = np.linspace(valores.min() - 3 * h, valores.max() + 3 * h, pontos)
    y = np.exp(-0.5 * ((x[:, None] - valores[None, :]) / h) ** 2).sum(axis=1)
    y = y / (len(valores) * h * np.sqrt(2 * np.pi))
    return x, y


# ---------------------------------------------------------------------------
# 2. Layout
# ---------------------------------------------------------------------------
app = Dash(_name_)
app.title = 'Dashboard E-commerce'

estilo_card = {'width': '48%', 'minWidth': '420px', 'display': 'inline-block',
               'verticalAlign': 'top', 'margin': '0.5%'}

app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f5f6fa'},
    children=[
        html.H1('Dashboard de Análise de E-commerce', style={'textAlign': 'center'}),
        html.P('Explore os produtos usando os filtros abaixo. Todos os gráficos são atualizados automaticamente.',
               style={'textAlign': 'center'}),

        # Filtros
        html.Div(
            style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px',
                   'marginBottom': '15px'},
            children=[
                html.Div(style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}, children=[
                    html.Label('Grupo de gênero', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='filtro-genero',
                                 options=[{'label': g, 'value': g} for g in GENEROS],
                                 value=GENEROS, multi=True),
                ]),
                html.Div(style={'width': '50%', 'display': 'inline-block', 'marginLeft': '4%',
                                'verticalAlign': 'top'}, children=[
                    html.Label('Faixa de preço (R$)', style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(id='filtro-preco', min=PRECO_MIN, max=PRECO_MAX, step=5,
                                    value=[PRECO_MIN, PRECO_MAX],
                                    marks={v: str(v) for v in range(PRECO_MIN, PRECO_MAX + 1, 50)},
                                    tooltip={'placement': 'bottom', 'always_visible': False}),
                ]),
                html.Div(id='resumo', style={'marginTop': '15px', 'fontWeight': 'bold'}),
            ],
        ),

        # Gráficos
        html.Div([
            html.Div(dcc.Graph(id='grafico-histograma'), style=estilo_card),
            html.Div(dcc.Graph(id='grafico-dispersao'), style=estilo_card),
            html.Div(dcc.Graph(id='grafico-mapa-calor'), style=estilo_card),
            html.Div(dcc.Graph(id='grafico-barras'), style=estilo_card),
            html.Div(dcc.Graph(id='grafico-pizza'), style=estilo_card),
            html.Div(dcc.Graph(id='grafico-densidade'), style=estilo_card),
            html.Div(dcc.Graph(id='grafico-regressao'), style=estilo_card),
        ]),
    ],
)


# ---------------------------------------------------------------------------
# 3. Callback: atualiza todos os gráficos a partir dos filtros
# ---------------------------------------------------------------------------
@app.callback(
    Output('resumo', 'children'),
    Output('grafico-histograma', 'figure'),
    Output('grafico-dispersao', 'figure'),
    Output('grafico-mapa-calor', 'figure'),
    Output('grafico-barras', 'figure'),
    Output('grafico-pizza', 'figure'),
    Output('grafico-densidade', 'figure'),
    Output('grafico-regressao', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-preco', 'value'),
)
def atualizar(generos, faixa_preco):
    generos = generos or []
    dff = df[(df['Genero_Grupo'].isin(generos)) &
             (df['Preço'].between(faixa_preco[0], faixa_preco[1]))]

    if dff.empty:
        titulos = ['Distribuição dos Preços', 'Preço x Desconto', 'Correlação entre Variáveis',
                   'Top 10 Marcas', 'Produtos por Gênero', 'Densidade de Preço', 'Regressão']
        return ('Nenhum produto encontrado.', *[figura_vazia(t) for t in titulos])

    resumo = (f'{len(dff)} produtos | Preço médio: R$ {dff["Preço"].mean():.2f} | '
              f'Nota média: {dff["Nota"].mean():.2f}')

    # 1) Histograma
    fig_hist = px.histogram(dff, x='Preço', nbins=25,
                            title='Distribuição dos Preços dos Produtos',
                            labels={'Preço': 'Preço (R$)'})
    fig_hist.update_layout(yaxis_title='Quantidade de produtos', bargap=0.05)

    # 2) Dispersão
    fig_disp = px.scatter(dff, x='Preço', y='Desconto', color='Genero_Grupo', opacity=0.7,
                          hover_data=['Título'],
                          title='Relação entre Preço e Desconto',
                          labels={'Preço': 'Preço (R$)', 'Desconto': 'Desconto (%)',
                                  'Genero_Grupo': 'Grupo'})

    # 3) Mapa de calor
    corr = dff[COLUNAS_NUM].corr().round(2)
    fig_calor = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1, aspect='auto',
                          title='Mapa de Calor: Correlação entre Variáveis Numéricas')
    fig_calor.update_layout(xaxis_title='Variáveis', yaxis_title='Variáveis')

    # 4) Barras: top 10 marcas
    top = dff['Marca_Limpa'].value_counts().head(10).sort_values().reset_index()
    top.columns = ['Marca', 'Quantidade']
    fig_barras = px.bar(top, x='Quantidade', y='Marca', orientation='h', text='Quantidade',
                        title='Top 10 Marcas com Mais Produtos')
    fig_barras.update_layout(xaxis_title='Quantidade de produtos', yaxis_title='Marca')

    # 5) Pizza
    contagem = dff['Genero_Grupo'].value_counts().reset_index()
    contagem.columns = ['Grupo', 'Quantidade']
    fig_pizza = px.pie(contagem, names='Grupo', values='Quantidade',
                       title='Distribuição dos Produtos por Grupo de Gênero')

    # 6) Densidade (um traço por grupo com pelo menos 2 produtos)
    fig_dens = go.Figure()
    for grupo, parte in dff.groupby('Genero_Grupo'):
        x, y = curva_densidade(parte['Preço'])
        if x is not None:
            fig_dens.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy',
                                          name=grupo, opacity=0.6))
    fig_dens.update_layout(title='Densidade de Preço por Grupo de Gênero',
                           xaxis_title='Preço (R$)', yaxis_title='Densidade')

    # 7) Regressão linear simples (numpy, sem depender de statsmodels)
    fig_reg = px.scatter(dff, x='Qtd_Vendidos_Cod', y='N_Avaliações', opacity=0.6,
                         title='Regressão: Quantidade Vendida x Número de Avaliações',
                         labels={'Qtd_Vendidos_Cod': 'Quantidade vendida (faixa em número)',
                                 'N_Avaliações': 'Número de avaliações'})
    if dff['Qtd_Vendidos_Cod'].nunique() > 1:
        a, b = np.polyfit(dff['Qtd_Vendidos_Cod'], dff['N_Avaliações'], 1)
        xs = np.linspace(dff['Qtd_Vendidos_Cod'].min(), dff['Qtd_Vendidos_Cod'].max(), 100)
        r = dff['Qtd_Vendidos_Cod'].corr(dff['N_Avaliações'])
        fig_reg.add_trace(go.Scatter(x=xs, y=a * xs + b, mode='lines',
                                     name=f'Regressão (R² = {r ** 2:.2f})',
                                     line=dict(color='red')))

    return resumo, fig_hist, fig_disp, fig_calor, fig_barras, fig_pizza, fig_dens, fig_reg


# ---------------------------------------------------------------------------
# 4. Execução
# ---------------------------------------------------------------------------
if _name_ == '_main_':
    app.run(debug=True)