import streamlit as st
import requests
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing, utils

# Define the SearXNG API URL (this could be a public instance or your own)
SEARXNG_API_URL = "https://searx.bndkt.io/search"  # Replace with your own SearXNG API URL

# Function to get SearXNG results based on the user input
def get_searxng_results(query):
    params = {
        "q": query,  # The search query
        "format": "json",  # JSON response format
        "categories": "general"  # Categories like 'general', 'news', etc. You can modify based on use case
    }
    try:
        response = requests.get(SEARXNG_API_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # We extract the top 3 results (you can modify this based on the data)
        search_results = data.get("results", [])
        if not search_results:
            return "Sorry, I couldn't find anything relevant."
        
        # For simplicity, we take the top 3 search results
        result_text = "\n".join([f"{result.get('title', '')}: {result.get('url', '')}" for result in search_results[:3]])
        return result_text
    except requests.exceptions.RequestException as e:
        return f"Error fetching search results: {e}"

# Define the chatbot model (Seq2Seq architecture - Encoder-Decoder) with training
def build_chatbot_model(vocab_size, embedding_dim, hidden_units):
    # Encoder
    encoder_input = layers.Input(shape=(None,))
    encoder_embedding = layers.Embedding(vocab_size, embedding_dim)(encoder_input)
    encoder_lstm = layers.LSTM(hidden_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_input = layers.Input(shape=(None,))
    decoder_embedding = layers.Embedding(vocab_size, embedding_dim)(decoder_input)
    decoder_lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = layers.Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_lstm_outputs)

    # Build and compile the model
    model = models.Model([encoder_input, decoder_input], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess text and create tokenizers
def preprocess_text(input_texts, target_texts, num_samples=10000):
    input_texts = input_texts[:num_samples]
    target_texts = ['\t' + text + '\n' for text in target_texts[:num_samples]]

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(input_texts + target_texts)
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    max_encoder_seq_length = max([len(seq) for seq in input_sequences])
    max_decoder_seq_length = max([len(seq) for seq in target_sequences])

    encoder_input_data = preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
    decoder_input_data = preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_decoder_seq_length, padding='post')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, len(tokenizer.word_index) + 1), dtype='float32')

    for i, seq in enumerate(target_sequences):
        for t, word_index in enumerate(seq):
            if t > 0:
                decoder_target_data[i, t - 1, word_index] = 1.0
    
    return tokenizer, encoder_input_data, decoder_input_data, decoder_target_data, max_encoder_seq_length, max_decoder_seq_length

# Function to save new query-response pairs (learning interaction) - optional feature
def save_query_response(query, response, file="chatbot_memory.json"):
    try:
        memory = {}
        # Load previous memories if they exist
        try:
            with open(file, 'r') as f:
                memory = json.load(f)
        except FileNotFoundError:
            pass
        
        # Save new query-response pair
        memory[query] = response
        
        # Write back to the memory file
        with open(file, 'w') as f:
            json.dump(memory, f, indent=4)
        
        return True
    except Exception as e:
        return f"Error saving data: {e}"

# Streamlit UI
def main():
    st.title("MaxChat")
    st.write("The smart bot. Check information before you use them")

    # Text input for user to ask questions
    user_input = st.text_input("Ask a question:")

    if user_input:
        # Fetch relevant search results from SearXNG
        search_response = get_searxng_results(user_input)
        
        # Display the results
        st.write(f"Chatbot (from SearXNG): {search_response}")
        
        # Optionally save the new query-response pair to memory for future reference (if you want)
        save_query_response(user_input, search_response)

if __name__ == "__main__":
    main()
