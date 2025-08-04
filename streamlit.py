import streamlit as st 
from openai_client import call_openai_api

# Page config 
st.set_page_config(page_title="OpenAI Chatbot", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# Header
st.header("🧪 _o3 and o4-mini_ Chatbot", divider="gray")

# Initialiser la session state pour la mémoire de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar 
with st.sidebar:
    model = st.selectbox("Choisir un Modèle", ("o3", "o4-mini-2025-04-16"))

    st.badge("Pourquoi essayer ce chatbot ?", color="blue")
    st.image("image/chart.png")
    
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.write(f"Messages en mémoire: {len(st.session_state.messages)}")

# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input 
user_input = st.chat_input(placeholder="Posez votre question", max_chars=10000)

# Traitement du message utilisateur
if user_input:
    # Ajouter le message utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Afficher le message utilisateur
    with st.chat_message("user"):
        st.write(user_input)
    
    # Maintenir seulement les 5 derniers messages (10 avec les réponses)
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]
    
    # Préparer le contexte pour l'API
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
    
    # Afficher la réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Le modèle réfléchit..."):
            response_placeholder = st.empty()
            reasoning_placeholder = st.empty()
            full_response = ""
            reasoning_text = ""
            
            try:
                # Appeler l'API OpenAI avec streaming
                for chunk in call_openai_api(context, model):
                    if isinstance(chunk, dict):
                        # Nouveau format avec types
                        if chunk.get("type") == "reasoning":
                            # Afficher le raisonnement
                            reasoning_text += chunk["content"]
                            with reasoning_placeholder.container():
                                st.info("🧠 **Raisonnement du modèle:**")
                                st.write(reasoning_text)
                        
                        elif chunk.get("type") == "text":
                            # Afficher la réponse
                            full_response += chunk["content"]
                            response_placeholder.write(full_response)
                        
                        elif chunk.get("type") == "metadata":
                            # Métadonnées finales
                            if chunk.get("sources") and chunk["sources"]:
                                st.write("\n**Sources:**")
                                for i, source in enumerate(chunk["sources"]):
                                    st.write(f"{i+1}. [{source['title']}]({source['url']})")
                            
                            
                            # Afficher les prix au lieu des tokens
                            if chunk.get("price_info") and chunk["price_info"].get("total_cost", 0) > 0:
                                price_info = chunk["price_info"]
                                st.success(f"💰 **Coût:** ${price_info['total_cost']:.6f} USD (Entrée: ${price_info['input_cost']:.6f}, Sortie: ${price_info['output_cost']:.6f})")
                            elif chunk.get("usage_info") and chunk["usage_info"]:
                                usage = chunk["usage_info"]
                                if 'input_tokens' in usage and 'output_tokens' in usage:
                                    st.caption(f"📊 Tokens - Entrée: {usage['input_tokens']}, Sortie: {usage['output_tokens']}")
                    
                    # Ancien format (rétrocompatibilité)
                    elif "sources" in chunk or "usage_info" in chunk:
                        if chunk.get("sources"):
                            st.write("\n**Sources:**")
                            for i, source in enumerate(chunk["sources"]):
                                st.write(f"{i+1}. [{source['title']}]({source['url']})")
                        
                        if chunk.get("price_info") and chunk["price_info"].get("total_cost", 0) > 0:
                            price_info = chunk["price_info"]
                            st.success(f"💰 **Coût:** ${price_info['total_cost']:.6f} USD")
                        elif chunk.get("usage_info"):
                            st.caption(f"📊 Tokens utilisés: {chunk['usage_info']}")
                
                else:
                    # Fragment de texte (ancien format)
                    if isinstance(chunk, str):
                        full_response += chunk
                        response_placeholder.write(full_response)
            
            except Exception as e:
                st.error(f"Erreur: {e}")
                full_response = f"Erreur lors de l'appel à l'API: {e}"
    
    # Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})
