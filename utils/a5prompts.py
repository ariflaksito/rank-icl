import pandas as pd
import numpy as np

def rank_sbert(embedding, data, topk = 3):
    df_data = pd.DataFrame(data)

    scores = []
    for _, row in df_data.iterrows(): # Iterate over DataFrame rows
        score = row['embedding'].dot(embedding)
        scores.append(score)

    df_data['score'] = scores
    return df_data.sort_values(by=['score'], ascending=False).head(topk).to_dict(orient='records')

def rank_bm25(doc, bm25, data, topk = 3):
    tokenized_query = doc.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[-topk:][::-1]
    df_data = pd.DataFrame(data).iloc[top_n_indices]
    return df_data.to_dict(orient='records')

def rank_tfidf(doc, vectorizer, tfidf_matrix, data, topk = 3):
        doc_vector = vectorizer.transform([doc])
        cosine_similarities = (tfidf_matrix @ doc_vector.T).toarray().flatten()
        top_n_indices = np.argsort(cosine_similarities)[-topk:][::-1]
        df_data = pd.DataFrame(data).iloc[top_n_indices]
        return df_data.to_dict(orient='records')

# function random selection
def rank_random(data, topk=3):
    df_data = pd.DataFrame(data)
    sampled_df = df_data.sample(n=topk, random_state=82)
    return sampled_df.to_dict(orient='records')

# wikisa prompt construction
def wiki_zero_prompt(query, doc):
    prompt = "You are an AI assistant that explain how query and document are related.\n"
    prompt += "Provide a few words of aspect explanation max 5 words.\n"
    prompt += "query: {query}.\n"
    prompt += "document: {doc}.\n"
    prompt += "aspect: "
    return prompt.format(query=query, doc=doc)

def wiki_few_prompt(query, doc, res):
    prompt = "You are an AI assistant that explain how query and document are related.\n"
    prompt += "I give you 3 examples below:\n"

    for i in range(len(res)):
        prompt += f"Example {i+1}\n"
        prompt += f"query: {res[i]['query']}.\n"
        prompt += f"document: {res[i]['doc']}\n"
        prompt += f"aspect: {res[i]['explanation']}.\n"

    prompt += "Provide a few words of aspect explanation max 5 words.\n"
    prompt += f"query: {query}.\n"
    prompt += f"document: {doc}.\n"
    prompt += "aspect: "
    return prompt

def wiki_rag_prompt(query, doc, res):

    prompt = "You are an AI assistant that explain how query and document are related.\n"
    prompt += "I give you 3 examples below:\n"

    for i in range(len(res)):   
            prompt += f"Example {i+1}\n"
            prompt += f"query: {res[i]['query']}.\n"
            prompt += f"document: {res[i]['doc']}\n"
            prompt += f"aspect: {res[i]['explanation']}.\n"

    prompt += "Provide a few words of aspect explanation max 5 words.\n"
    prompt += f"query: {query}.\n"
    prompt += f"document: {doc}.\n"
    prompt += "aspect: "
    return prompt

# exarank prompt construction
def exa_zero_prompt(query, doc):
    prompt = "You are an AI assistant that explain how query and document are related.\n"
    prompt += "Provide an explanation if the query is related to the document.\n"
    prompt += "query: {query}.\n"
    prompt += "document: {doc}.\n"
    prompt += "explanation: "
    return prompt.format(query=query, doc=doc)

def exa_few_prompt(query, doc, res):
    prompt = "You are an AI assistant that explain how query and document are related.\n"
    prompt += "I give you 3 examples below:\n"

    for i in range(len(res)):
        prompt += f"Example {i+1}\n"
        prompt += f"query: {res[i]['query']}.\n"
        prompt += f"document: {res[i]['doc']}\n"
        prompt += f"explanation: {res[i]['explanation']}.\n"

    prompt += "Provide an explanation if the query is related to the document.\n"
    prompt += f"query: {query}.\n"
    prompt += f"document: {doc}\n"
    prompt += "explanation: "
    return prompt

def exa_rag_prompt(query, doc, res):
    prompt = "You are an AI assistant that explain how query and document are related.\n"
    prompt += "I give you 3 examples below:\n"

    for i in range(len(res)):
        prompt += f"Example {i+1}\n"
        prompt += f"query: {res[i]['query']}.\n"
        prompt += f"document: {res[i]['doc']}\n"
        prompt += f"explanation: {res[i]['explanation']}.\n"

    prompt += "Provide an explanation if the query is related to the document.\n"
    prompt += f"query: {query}.\n"
    prompt += f"document: {doc}\n"
    prompt += "explanation: "
    return prompt
