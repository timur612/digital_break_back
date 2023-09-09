import numpy as np
import json
import string

# import nltk
#
# nltk.download("punkt")
# nltk.download("wordnet")
#
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from gensim.models import Word2Vec


def clean_text(text):
    table = text.maketrans(dict.fromkeys(string.punctuation))

    words = word_tokenize(text.lower().strip().translate(table))
    # words = [word for word in words if word not in stopwords.words('russian')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]
    return " ".join(lemmed)


def file_2_vectors(file) -> tuple:
    """

    :params file: json-file
    :return embeddings list, Word2Vec model:
    """
    answers = []
    with open(file, encoding="utf_8") as f:
        q_a = json.loads(f.read())
        q = q_a["question"]
        # print(q)
        for a in q_a["answers"]:
            answers.append([clean_text(a["answer"] + " " + q)])

    # corpus = []
    # for sentence in answers:
    #     corpus.append(sentence[0].split())
    model = Word2Vec(answers, vector_size=100, min_count=1)
    # print(answers)
    vectors = []
    for sentence in answers:
        if len(sentence[0]) == 0:
            word_vec = [0] * 100
        elif sentence[0] not in model.wv.key_to_index.keys():
            w_vs = []
            for word in sentence[0].split():
                w_vs.append(model.wv[word])
            word_vec = np.mean(np.array(w_vs), axis=0)
        else:
            word_vec = model.wv[sentence[0]]
        vectors.append(word_vec)
    return vectors, model


if __name__ == "__main__":
    vectors, model = (file_2_vectors("../../data/1704.json"))
    print(len(vectors))
