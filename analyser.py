import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import praw
from collections import Counter
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import time


def analisar_perfil_usuario(username, reddit, model, vectorizer, limite=100):
    def avaliar_toxicidade_local(comentario):
        comentario_processado = comentario.lower()
        comentario_vect = vectorizer.transform([comentario_processado])
        predicao = model.predict(comentario_vect)
        probabilidades = model.predict_proba(comentario_vect)
        prob_odio = probabilidades[0][1]
        
        return {
            "odio": predicao[0] == 1,
            "nivel_toxicidade": prob_odio
        }
    
    try:
        user = reddit.redditor(username)
        try:
            user_created = user.created_utc
        except Exception as e:
            return {'erro': f'Usuário {username} não encontrado', 'detalhes': str(e)}

        comentarios_analisados = 0
        total_toxicidade = 0
        comentarios_odiosos = 0
        historico_comentarios = []
        subreddits_atividade = []
        comentarios_por_dia = {}

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
            if comentarios_analisados % 10 == 0:
                time.sleep(0.1)

        if comentarios_analisados == 0:
            return {'erro': f'Usuário {username} não possui comentários públicos'}

        nivel_medio_toxicidade = total_toxicidade / comentarios_analisados
        percentual_odioso = (comentarios_odiosos / comentarios_analisados) * 100
        subreddits_counter = Counter(subreddits_atividade)
        top_subreddits = subreddits_counter.most_common(10)
        comentarios_mais_toxicos = sorted(historico_comentarios, key=lambda x: x['toxicidade'], reverse=True)[:5]

        return {
            'username': username,
            'resumo': {
                'total_comentarios_analisados': comentarios_analisados,
                'comentarios_odiosos': comentarios_odiosos,
                'percentual_odioso': round(percentual_odioso, 2),
                'nivel_medio_toxicidade': round(nivel_medio_toxicidade, 4),
            },
            'atividade': {
                'subreddits_mais_ativos': top_subreddits,
                'comentarios_por_dia': comentarios_por_dia
            },
            'comentarios_mais_toxicos': comentarios_mais_toxicos
        }
    except Exception as e:
        return {'erro': f'Erro inesperado ao analisar {username}', 'detalhes': str(e)}