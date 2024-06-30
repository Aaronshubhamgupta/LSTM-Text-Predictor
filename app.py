
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



# Load the dataset
df = pd.read_csv('/content/medium_data.csv')

# Combine titles and subtitles into a single text column
df['text'] = df['title'].fillna('') + ' ' + df['subtitle'].fillna('')

# Tokenization and preparation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
total_words = len(tokenizer.word_index) + 1

# Create input sequences using tokenized texts
input_sequences = []
for line in df['text']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences for equal input length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]

# Convert labels to categorical
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(xs, ys, epochs=1, batch_size=32, verbose=1)

# Function to generate next word prediction
def generate_next_word(seed_text, next_words, tokenizer, max_sequence_len, model):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Get the index of the word with maximum probability
        predicted_index = np.argmax(predicted_probs)

        # Find the word corresponding to the index
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        seed_text += " " + predicted_word

    return seed_text

# Streamlit App
st.title('Text Generation with LSTM Model')

# Input text box for seed text
seed_text = st.text_input('Enter Seed Text', 'A Beginner’s Guide to')

# Slider for number of words to predict
next_words = st.slider('Number of Words to Generate', 1, 50, 5)

# Generate button
if st.button('Generate'):
    predicted_text = generate_next_word(seed_text, next_words, tokenizer, max_sequence_len, model)
    st.write(predicted_text)