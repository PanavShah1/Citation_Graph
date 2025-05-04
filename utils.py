import pickle
import networkx as nx
import torch
import os 
import random

def load_data():
    G = nx.read_graphml('data/citation_graph.graphml')
    with open("data/papers.pkl", 'rb') as f:
        papers = pickle.load(f)
    return G, papers

def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, 
                       return_tensors="pt",
                       padding='max_length',
                       truncation=True,
                       max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state

    # mean pooling
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_outputs = outputs * attention_mask
    mean_pooled = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1)
    return mean_pooled.squeeze(0)


def embed_all_papers(papers, tokenizer, tokmodel, device, cache_path='data/paper_embeddings.pkl'):
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            paper_embeddings = pickle.load(f)
        return {k: v.to(device) for k, v in paper_embeddings.items()}
    else:
        paper_embeddings = {}
        for i in range(len(papers)):
            paper = papers[i]
            title = paper['title']
            abstract = paper['abstract']
            text = title + '\n\n' + abstract
            embedding = embed_text(text, tokenizer, tokmodel)
            paper_embeddings[paper['index']] = embedding
        
        with open(cache_path, "wb") as f:
            pickle.dump(paper_embeddings, f)

        return paper_embeddings
