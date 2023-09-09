# from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoConfig

# import numpy as np
# from scipy.special import softmax

# import json
# from translate import Translator

# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)

# sentiment_dict = {
#                     0: 'Negative',
#                     1: "Neutral",
#                     2: 'Positive'
#                 }


# translator = Translator(from_lang="russian",to_lang="english")
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# def json_to_sentiment(filename):
#     list_of_sentiment = []

#     with open(filename,  encoding="utf-8") as f:
#         json_file = json.loads(f.read())

#     for answer in json_file['answers']:
#         text_answer = answer['answer']
#         list_of_sentiment.append(get_sent(text_answer))
        
#     return list_of_sentiment
    
# def get_sent(text):
#     try:
#         text = translator.translate(text)
#         encoded_input = tokenizer(text, return_tensors='pt')
#         output = model(**encoded_input)
#         scores = output[0][0].detach().numpy()
#         scores = softmax(scores)
#         label = sentiment_dict[scores.argmax()]
#     except:
#         label='Neutral'
#     return label
