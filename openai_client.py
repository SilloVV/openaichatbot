import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API. Assurez-vous que votre fichier .env contient OPENAI_API_KEY="..."
api_key = os.getenv("OPENAI_API_KEY")

# Initialiser le client OpenAI
client = OpenAI(api_key=api_key)

def calculate_price(model: str, input_tokens: int, output_tokens: int) -> dict:
    pricing = {
        "o3": {"input": 2.0, "output": 8.0},
        "o4-mini": {"input": 0.55, "output": 2.20},
        "o4-mini-2025-04-16": {"input": 0.55, "output": 2.20}
    }
    
    if model not in pricing:
        return {"input_cost": 0, "output_cost": 0, "total_cost": 0, "currency": "USD"}
    
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6), 
        "total_cost": round(total_cost, 6),
        "currency": "USD"
    }

def call_openai_api(user_input: str, model: str, max_tokens: int = 5000, show_reasoning: bool = True):
    # Détecter si c'est une question juridique et ajouter instruction de recherche web
    juridique_keywords = ["code", "loi", "article", "juridique", "droit", "tribunal", "jurisprudence", "décret", "ordonnance", "réglementation", "légal","conforme", "confirmité"]
    is_juridique = any(keyword in user_input.lower() for keyword in juridique_keywords)
    
    if is_juridique:
        enhanced_input = f"""INSTRUCTION IMPORTANTE: Cette question concerne le droit. Tu DOIS absolument effectuer une recherche internet systématique pour obtenir les informations les plus récentes et précises. Utilise tes outils de recherche web pour vérifier les sources officielles, codes, lois et jurisprudences actuelles avant de répondre.

Question de l'utilisateur: {user_input}"""
    else:
        enhanced_input = user_input
    
    # Configurer les paramètres pour afficher le raisonnement si demandé
    request_params = {
        "model": model,
        "tools": [{"type": "web_search_preview"}],
        "input": enhanced_input,
        "max_output_tokens": max_tokens,
        "stream": True,
    }
    

    
    response_stream = client.responses.create(**request_params)
    
    # --- Variables pour stocker les informations extraites ---
    full_response_text = ""
    reasoning_text = ""
    sources = []
    usage_info = {}
    has_reasoning = False
    spinner_container = None
    
    # Créer un spinner Streamlit si on est dans un contexte Streamlit
    try:
        if not has_reasoning:
            spinner_container = st.empty()
            with spinner_container:
                st.spinner("🤔 Le modèle réfléchit...")
    except:
        pass


    for chunk in response_stream:
        # Traiter le raisonnement du modèle
        if hasattr(chunk, 'type') and chunk.type == 'response.reasoning.delta':
            try:
                reasoning_text += chunk.delta
                has_reasoning = True
                # Effacer le spinner et afficher le raisonnement
                if spinner_container:
                    spinner_container.empty()
                    spinner_container = None
                yield {"type": "reasoning", "content": chunk.delta}
            except AttributeError:
                pass
        
        # Traiter les événements de texte (les 'deltas' sont les fragments de texte)
        elif hasattr(chunk, 'type') and chunk.type == 'response.output_text.delta':
            try:
                # Effacer le spinner si on commence à recevoir la réponse
                if spinner_container:
                    spinner_container.empty()
                    spinner_container = None
                # Concaténer le texte au fur et à mesure
                full_response_text += chunk.delta
                # Afficher le texte en temps réel, comme un chatbot
                yield {"type": "text", "content": chunk.delta}
            except AttributeError:
                print("Erreur lors de la concaténation du texte. (ligne 112)")
        
        # Traiter les événements d'annotation pour récupérer les sources
        elif hasattr(chunk, 'type') and chunk.type == 'response.output_text.annotation.added':
            annotation = chunk.annotation
            if annotation['type'] == 'url_citation':
                # Stocker les informations pertinentes de la source
                sources.append({
                    "title": annotation['title'],
                    "url": annotation['url']
                })
            
        elif hasattr(chunk, 'response') and hasattr(chunk.response, 'usage'):
            try:
                usage_info = chunk.response.usage.model_dump()
            except AttributeError:
                pass

    # Effacer le spinner s'il est encore là
    if spinner_container:
        spinner_container.empty()
    
    # Calculer les prix si on a les informations d'utilisation
    price_info = {}
    if usage_info and 'input_tokens' in usage_info and 'output_tokens' in usage_info:
        price_info = calculate_price(model, usage_info['input_tokens'], usage_info['output_tokens'])
    
    # Yielder les métadonnées à la fin
    yield {"type": "metadata", "sources": sources, "usage_info": usage_info, "price_info": price_info, "full_response_text": full_response_text, "reasoning_text": reasoning_text}

if __name__ == "__main__":
    sources = []
    usage_info = {}
    price_info = {}
    full_response_text = ""
    
    for chunk in call_openai_api("Que dit l'article L.121-2 du code de commerce français ?", "o3"):
        if isinstance(chunk, dict):
            # C'est le dictionnaire de métadonnées à la fin
            sources = chunk["sources"]
            usage_info = chunk["usage_info"]
            price_info = chunk.get("price_info", {})
            print(chunk, end="", flush=True)
        else:
            # C'est un fragment de texte
            print(chunk, end="", flush=True)
    
    # Afficher les prix au lieu des tokens
    if price_info and price_info.get("total_cost", 0) > 0:
        print(f"\n\nCoût de la requête :")
        print(f"  - Coût des tokens d'entrée : ${price_info['input_cost']:.6f}")
        print(f"  - Coût des tokens de sortie : ${price_info['output_cost']:.6f}")
        print(f"  - Coût total : ${price_info['total_cost']:.6f} {price_info['currency']}")
    elif usage_info:
        print(f"\n\nInformations sur l'utilisation :")
        if 'input_tokens' in usage_info:
            print(f"  - Tokens d'entrée : {usage_info['input_tokens']}")
        if 'output_tokens' in usage_info:
            print(f"  - Tokens de sortie : {usage_info['output_tokens']}")
    
    # Afficher les sources
    if sources:
        print("\n\nSources (citations) :\n")
        for i, source in enumerate(sources):
            print(f"{i+1}. Titre : {source['title']}")
            print(f"   URL   : {source['url']}")
    else:
        print("\n\nAucune source n'a été trouvée.")