from configparser import SectionProxy
from json import tool
from gensim.models import KeyedVectors
import random
from numpy import negative
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from PyDictionary import PyDictionary


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


# ---
chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    # api_key="" # Optional if not set as an environment variable
)

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

@tool
def get_synonyms():
    dictionary = PyDictionary()
    return dictionary.synonym(secret_choice)

@tool
def get_antonyms():
    dictionary = PyDictionary()
    return dictionary.antonym(secret_choice)

@tool
def get_word_starts_with():
    return secret_choice[0]


while currentWord != secret_choice:
    currentWord = input('Enter a word: ')
    try:   
        print('The similarity between the word you entered and the secret word is: ', model.similarity(currentWord, secret_choice))
    except KeyError:
        print(f'The word {currentWord} is not in the vocabulary')
        continue
    word_list.add(currentWord)
    ranked_list = sorted(list(word_list), key=lambda x: model.similarity(x, secret_choice), reverse=True)
    if (len(ranked_list) > 10):
        print(secret_choice)
        break
    for i, word in enumerate(ranked_list, 1):
        print(f"{i}. {word}")
    print()
    if (len(ranked_list) > 5):
        print(model.most_similar(positive=ranked_list[0:5]+[secret_choice], topn=10))
        print(model.doesnt_match(ranked_list))
        print(model.most_similar(negative=[ranked_list[0]], positive=[secret_choice], topn=5))