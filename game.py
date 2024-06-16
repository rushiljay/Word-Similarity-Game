from configparser import SectionProxy
from gensim.models import KeyedVectors
import random
from numpy import negative
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from PyDictionary import PyDictionary
import streamlit as st


model_path = 'GoogleNews-vectors-negative300.bin'
noun_list_link = 'https://www.desiquintans.com/downloads/nounlist/nounlist.txt'

model = KeyedVectors.load_word2vec_format(model_path, binary=True)
response = requests.get(noun_list_link, timeout=5)
vocabulary = response.text.split('\n')

def get_word():
    choice = int(random.random()*(len(vocabulary)-1))
    if vocabulary[choice] in model:
        return vocabulary[choice]
    else:
        return get_word()

# if word not in vocabulary check, and account for spaces

secret_choice = get_word()

word_list = set()

currentWord = ''

ranked_list = []

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

dictionary = PyDictionary()

@tool
def get_synonyms():
    """Returns a list of synonyms for the secret word."""
    return dictionary.synonym(secret_choice)

@tool
def get_antonyms():
    """Returns a list of antonyms for the secret word."""
    return dictionary.antonym(secret_choice)

@tool
def get_word_definition():
    """Returns the definition of the secret word."""
    return dictionary.meaning(secret_choice)

@tool
def get_word_starts_with():
    """Returns the first letter of the secret word."""
    return secret_choice[0]

@tool
def get_word_ends_with():
    """Returns the last letter of the secret word."""
    return secret_choice[-1]

@tool
def get_doesnt_match():
    """Returns the word that doesn't match the secret word in the ranked list."""
    return model.doesnt_match(ranked_list)

@tool
def get_word_length():
    """Returns the length of the secret word."""
    return len(secret_choice)

@tool  
def get_word():
    """Returns the secret word."""
    return secret_choice


if __name__ == '__main__':

    with st.sidebar:
        groq_api_key = st.text_input("Groq API Key", key="groq_api_key", type="password")

    # Streamlit page setup
    st.title('Semantic Word Game')
    st.write("Insert a word and I'll tell you how similar it is to the secret word. The secret word is a noun.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can help you find the secret word. How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    # if "messages" not in st.session_state:
    #     st.session_state["messages"] = [
    #         {"role": "assistant", "content": "Hi, I'm a chatbot who can give you stock market data and perform research. How can I help you?"}
    #     ]

    # for msg in st.session_state.messages:
    #     st.chat_message(msg["role"]).write(msg["content"])

    IN_PROGRESS = False

    if prompt := st.chat_input(placeholder="Type a word here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if IN_PROGRESS:
            st.info("Please wait for the current request to finish.")
            st.stop()
        else:
            IN_PROGRESS = True
            st.chat_message("user").write(prompt)

        print(IN_PROGRESS)

        if not groq_api_key:
            st.info("Please add your Groq API key to continue.")
            st.stop()

        llm = ChatGroq(temperature=0.5, model_name="llama3-70b-8192", groq_api_key=groq_api_key) # plz dont change

        tools = [get_synonyms, get_antonyms, get_word_definition, get_word_starts_with, get_word_ends_with, get_doesnt_match]

        word_list.add(currentWord)
        ranked_list = sorted(list(word_list), key=lambda x: model.similarity(x, secret_choice), reverse=True)

        system_message = "You are a helpful assistant."

        with open('system.txt', 'r') as file:
            system_message = file.read()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )    
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        chain = agent_executor

        with st.chat_message("assistant"):
            #print(st.session_state.messages[-1])
            response = ''
            i = 0
            REATTEMPT_LIMIT = 3
            query = st.session_state.messages[-1]['content']
            while i < REATTEMPT_LIMIT:
                print(query, i)
                try:
                    if (IN_PROGRESS):
                        tempQuery = query
                        response = chain.invoke({"input": tempQuery})['output']
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        response = response.replace("$","\$")


                        # for word in response.split(" "):
                        #     time.sleep(0.01)
                        #     st.write_stream(word + " ")

                        def stream_data():
                            for word in response.split(" "):
                                    yield word + " "
                                    time.sleep(0.02)
                        
                        st.write_stream(stream_data)
                        #st.write(response)
                        IN_PROGRESS = False
                        i = 0
                    break
                except Exception as e:
                    query = "USE LESS TOOLS!!! Don't use \"get_news\"!!! " + query
                    #tools = tools[1:-1]
                    print(e)
                    response = e
                    i += 1