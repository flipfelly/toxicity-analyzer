import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import praw
from collections import Counter
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import time

# ======================================================
# Configuração da página
# ======================================================
st.set_page_config(
    page_title="Toxicity Analyzer • Reddit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS com design glassmorphism minimalista
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Reset e tema base */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0505 100%);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal com glassmorphism */
    .main-header {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #ef4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: white;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.875rem;
        font-weight: 400;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Métricas com glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.75rem 1.25rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #fff;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .metric-toxic .metric-value {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Badge de classificação */
    .classification-badge {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .badge-label {
        display: inline-block;
        padding: 0.75rem 2rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .extreme { 
        background: linear-gradient(135deg, #dc2626, #991b1b); 
        color: #fff;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .high { 
        background: linear-gradient(135deg, #ea580c, #c2410c); 
        color: #fff;
    }
    
    .moderate { 
        background: linear-gradient(135deg, #f59e0b, #d97706); 
        color: #fff;
    }
    
    .low { 
        background: linear-gradient(135deg, #10b981, #059669); 
        color: #fff;
    }
    
    .clean { 
        background: linear-gradient(135deg, #06b6d4, #0891b2); 
        color: #fff;
    }
    
    /* Comentários tóxicos */
    .toxic-comment {
        background: rgba(239, 68, 68, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(239, 68, 68, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .toxic-comment:hover {
        background: rgba(239, 68, 68, 0.05);
        transform: translateX(4px);
    }
    
    .rank-number {
        position: absolute;
        top: -8px;
        left: 20px;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: #fff;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.875rem;
    }
    
    .comment-text {
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    .comment-meta {
        display: flex;
        gap: 1.5rem;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .toxicity-indicator {
        width: 100%;
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        overflow: hidden;
        margin: 0.75rem 0;
    }
    
    .toxicity-fill {
        height: 100%;
        background: linear-gradient(90deg, #dc2626, #ef4444);
        border-radius: 2px;
    }
    
    /* Botão customizado */
    .stButton > button {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #fff !important;
        padding: 0.875rem 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(239, 68, 68, 0.4) !important;
        box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.2) !important;
    }
    
    /* Seções */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(239, 68, 68, 0.3), transparent);
        margin: 3rem 0 2rem 0;
    }
    
    .section-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ef4444, #dc2626) !important;
    }
    /* SLIDER */
     .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ef4444, #dc2626) !important;
    }
    
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 100px !important;
    }
    
    .stSlider > div > div > div > div > div {
    
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# Função de análise
# ======================================================
def analisar_perfil_usuario(username, reddit, model, vectorizer, limite=100):
    def avaliar_toxicidade_local(comentario):
        try:
            comentario_vect = vectorizer.transform([comentario.lower()])
            predicao = model.predict(comentario_vect)
            probabilidades = model.predict_proba(comentario_vect)
            return {
                "odio": predicao[0] == 1,
                "nivel_toxicidade": probabilidades[0][1]
            }
        except:
            return {"odio": False, "nivel_toxicidade": 0}
    
    try:
        user = reddit.redditor(username)
        try:
            _ = user.created_utc
        except:
            return {'erro': f'Usuário {username} não encontrado'}

        comentarios_analisados = 0
        total_toxicidade = 0
        comentarios_odiosos = 0
        historico_comentarios = []
        subreddits_atividade = []
        comentarios_por_dia = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for comment in user.comments.new(limit=limite):
            if comentarios_analisados >= limite:
                break
            if not comment.body or comment.body in ['[deleted]', '[removed]']:
                continue

            resultado = avaliar_toxicidade_local(comment.body)
            total_toxicidade += resultado["nivel_toxicidade"]

            if resultado["odio"]:
                comentarios_odiosos += 1

            data_comentario = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
            data_str = data_comentario.strftime('%Y-%m-%d')
            comentarios_por_dia[data_str] = comentarios_por_dia.get(data_str, 0) + 1
            subreddits_atividade.append(comment.subreddit.display_name)

            historico_comentarios.append({
                'texto': comment.body[:200] + "..." if len(comment.body) > 200 else comment.body,
                'toxicidade': resultado["nivel_toxicidade"],
                'odio': resultado["odio"],
                'subreddit': comment.subreddit.display_name,
                'data': data_str,
                'score': comment.score,
                'url': f"https://reddit.com{comment.permalink}"
            })

            comentarios_analisados += 1
            progress = min(comentarios_analisados / limite, 1.0)
            progress_bar.progress(progress)
            status_text.text(f'Analisando... {comentarios_analisados}/{limite}')

        progress_bar.empty()
        status_text.empty()

        if comentarios_analisados == 0:
            return {'erro': 'Nenhum comentário público encontrado'}

        percentual_odioso = (comentarios_odiosos / comentarios_analisados) * 100
        subreddits_counter = Counter(subreddits_atividade)
        
        return {
            'username': username,
            'resumo': {
                'total_comentarios_analisados': comentarios_analisados,
                'comentarios_odiosos': comentarios_odiosos,
                'percentual_odioso': round(percentual_odioso, 2),
                'nivel_medio_toxicidade': round(total_toxicidade / comentarios_analisados * 100, 2),
            },
            'atividade': {
                'subreddits_mais_ativos': subreddits_counter.most_common(10),
                'comentarios_por_dia': comentarios_por_dia
            },
            'comentarios_mais_toxicos': sorted(historico_comentarios, 
                                              key=lambda x: x['toxicidade'], 
                                              reverse=True)[:5]
        }
    except Exception as e:
        return {'erro': f'Erro: {str(e)}'}

def get_classification(percentual):
    if percentual >= 70:
        return "Extremamente Tóxico", "extreme"
    elif percentual >= 50:
        return "Altamente Tóxico", "high"
    elif percentual >= 25:
        return "Moderadamente Tóxico", "moderate"
    elif percentual >= 10:
        return "Levemente Tóxico", "low"
    else:
        return "Perfil Limpo", "clean"

# ======================================================
# Inicialização
# ======================================================
@st.cache_resource
def init_reddit_and_models():
    load_dotenv()
    
    try:
        model = joblib.load("modelo_odio.joblib")
        vectorizer = joblib.load("vetorizador_odio.joblib")
        
        reddit = praw.Reddit(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            user_agent=os.getenv("USER_AGENT"),
            check_for_async=False
        )
        
        return reddit, model, vectorizer
    except Exception as e:
        st.error(f"Erro na inicialização: {e}")
        st.stop()

reddit, model, vectorizer = init_reddit_and_models()

# ======================================================
# Interface
# ======================================================

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">TOXICITY ANALYZER</h1>
    <p class="main-subtitle">Reddit Profile Analysis</p>
</div>
""", unsafe_allow_html=True)

# Input
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    username = st.text_input("", placeholder="Digite o username", label_visibility="collapsed")
    
    #limite de comentários
    limite_comentarios = st.slider(
        "Limite de comentários para análise",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Defina quantos comentários serão analisados (máximo 500)"
    )
    
    analyze_button = st.button("ANALISAR", use_container_width=True)

if analyze_button and username:
    resultado = analisar_perfil_usuario(username, reddit, model, vectorizer, limite=limite_comentarios)


    if "erro" in resultado:
        st.error(f"⚡ {resultado['erro']}")
    else:
        resumo = resultado["resumo"]
        classificacao, classe = get_classification(resumo['percentual_odioso'])
        
        # Métricas
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{resumo['total_comentarios_analisados']}</div>
                <div class="metric-label">Comentários</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-toxic">
                <div class="metric-value">{resumo['comentarios_odiosos']}</div>
                <div class="metric-label">Tóxicos</div>
            </div>
            """, unsafe_allow_html=True)
        

        # Classificação
        st.markdown(f"""
        <div class="classification-badge">
            <div class="badge-label {classe}">{classificacao}</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.875rem;">
                Usuário <span style="color: #ef4444;">u/{resultado['username']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gráficos
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Visualizações</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Tóxico', 'Normal'],
                values=[resumo["comentarios_odiosos"], 
                       resumo["total_comentarios_analisados"] - resumo["comentarios_odiosos"]],
                hole=.7,
                marker=dict(colors=['#ef4444', '#1f2937']),
                textfont=dict(color='white')
            )])
            
            fig_pie.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                annotations=[dict(
                    text=f'{resumo["percentual_odioso"]}%',
                    x=0.5, y=0.5,
                    font_size=32,
                    font_color='#ef4444',
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Subreddits
            if resultado["atividade"]["subreddits_mais_ativos"]:
                subs_df = pd.DataFrame(resultado["atividade"]["subreddits_mais_ativos"][:5], 
                                      columns=["Subreddit", "Count"])
                
                fig_bar = go.Figure(go.Bar(
                    x=subs_df["Count"],
                    y=subs_df["Subreddit"],
                    orientation='h',
                    marker=dict(color='#ef4444'),
                    text=subs_df["Count"],
                    textposition='outside',
                    textfont=dict(color='white')
                ))
                
                fig_bar.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(visible=False),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    margin=dict(t=0, b=0)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        if resultado["atividade"]["comentarios_por_dia"]:
            
            # Preparar dados para o gráfico de linha
            atividade_df = pd.DataFrame(list(resultado["atividade"]["comentarios_por_dia"].items()), 
                                      columns=["Data", "Comentários"])
            atividade_df["Data"] = pd.to_datetime(atividade_df["Data"])
            atividade_df = atividade_df.sort_values("Data")
            
            # Criar gráfico de linha
            fig_line = go.Figure()
            
            fig_line.add_trace(go.Scatter(
                x=atividade_df["Data"],
                y=atividade_df["Comentários"],
                mode='lines+markers',
                line=dict(color='#ef4444', width=3),
                marker=dict(
                    color='#ef4444',
                    size=8,
                    line=dict(color='#dc2626', width=2)
                ),
                fill='tonexty',
                fillcolor='rgba(239, 68, 68, 0.1)',
                name='Comentários por dia'
            ))
            
            fig_line.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)',
                    title='Data',
                    title_font=dict(color='rgba(255,255,255,0.7)', size=14)
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)',
                    title='Número de Comentários',
                    title_font=dict(color='rgba(255,255,255,0.7)', size=14)
                ),
                margin=dict(t=20, b=60, l=60, r=60),
                showlegend=False
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
        # Top comentários tóxicos
        if resultado["comentarios_mais_toxicos"]:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Comentários Mais Tóxicos</div>', unsafe_allow_html=True)
            
            for i, com in enumerate(resultado["comentarios_mais_toxicos"], 1):
                toxicity_percent = round(com['toxicidade'] * 100, 1)
                st.markdown(f"""
                <div class="toxic-comment">
                    <span class="rank-number">#{i}</span>
                    <div class="toxicity-indicator">
                        <div class="toxicity-fill" style="width: {toxicity_percent}%"></div>
                    </div>
                    <div class="comment-text">{com['texto']}</div>
                    <div class="comment-meta">
                        <span>r/{com['subreddit']}</span>
                        <span>{toxicity_percent}% tóxico</span>
                        <span>Score: {com['score']}</span>
                        <span>{com['data']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)