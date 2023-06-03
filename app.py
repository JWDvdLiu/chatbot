import streamlit as st
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
import json 

def clean_output(output_tokens):
    # Remove the <sos> token (if it's the first token)
    if tf.equal(output_tokens[0], 1):
        output_tokens = output_tokens[1:]

    # Find the index of the first <eos> token
    eos_positions = tf.where(tf.equal(output_tokens, 2))

    # If there's at least one <eos> token
    if tf.shape(eos_positions)[0] > 0:
        # Get the index of the first <eos> token
        eos_position = eos_positions[0, 0]

        # Remove <eos> and everything after it
        output_tokens = output_tokens[:eos_position]

    return output_tokens

# Load your trained model
@st.cache_resource()
def load_model_and_tokenizer():
    model = tf.saved_model.load('mymodel')  # Load your model here
    with open('inp_lang.json') as f:
        json_data = json.load(f)
        inp_tokenizer = tokenizer_from_json(json_data)
    
    with open('tar_lang.json') as f:
        json_data = json.load(f)
        tar_tokenizer = tokenizer_from_json(json_data)
        
    return model,inp_tokenizer,tar_tokenizer

def get_response(sentence,model, inp_tokenizer, tar_tokenizer):
    encode_input = inp_tokenizer.texts_to_sequences([sentence])
    encode_input = tf.keras.preprocessing.sequence.pad_sequences(encode_input,
                                                        padding='post')
    encode_input = tf.ragged.constant(encode_input, dtype=tf.int32).to_tensor()

    output = model(encode_input)
    output = clean_output(output[0]).numpy()
    response = tar_tokenizer.sequences_to_texts([output])
    response = ' '.join(response)
    return response

# Main function
def main():
    
    model, inp_tokenizer,tar_tokenizer = load_model_and_tokenizer()

    # Title
    st.title('My Transformer chatbot')
    
    conversation_history = st.empty()
    
    
    
    # User input
    user_input = st.text_input("Your message", "")
    
    # Placeholder to store the history of conversation
    messages = []
    if st.button('Send'):

        response = get_response(user_input, model, inp_tokenizer,tar_tokenizer)

        # Update the messages
        messages.append({"User": user_input, "Chatbot": response})

        # Keep the last 10 messages
        messages = messages[-10:]

        # Display the conversation history
        conversation_history.markdown('\n'.join([f'{k}: {v}' for m in messages for k, v in m.items()]))

        # Clear the text input
        st.empty()

if __name__ == '__main__':
    main()