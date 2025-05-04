import torch
import pickle
from utils import embed_text

def rank_papers(test_title, test_abstract, projection_mlp, tokenizer, embedder, link_predictor, device):
    test_text = test_title + "\n\n" + test_abstract
    test_emb = embed_text(test_text, tokenizer, embedder).to(device)

    with torch.no_grad():
        projected_emb = projection_mlp(test_emb.unsqueeze(0)).squeeze(0)

    with open('data/graphsage_embeddings.pkl', 'rb') as f:
        graphsage_embs = pickle.load(f)
    
    scores = []
    link_predictor.eval()

    for idx, emb in graphsage_embs.items():
        emb = emb.to(projected_emb.device)
        with torch.no_grad():
            score = link_predictor(projected_emb.unsqueeze(0), emb.unsqueeze(0)).item()
        scores.append((idx, score))
    
    ranked = sorted(scores, key = lambda x: -x[1])
    return [idx for idx, _ in ranked]


