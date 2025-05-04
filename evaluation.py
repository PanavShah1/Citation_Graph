import argparse
import os
import pickle
import networkx as nx

import torch
import torch_geometric
from torch_geometric.utils import from_networkx
from transformers import AutoTokenizer, AutoModel

from models import GraphSAGE, ProjectionMLP, LinkPredictor
from utils import load_data, embed_all_papers, split_data
from inference import rank_papers
from trainers import (
    train_GraphSAGE,
    train_projection_mlp,
    train_link_predictor
)


train_override = False # don't train again -> False

################################################
#               IMPORTANT                      #
################################################
# 1. Do not print anything other than the ranked list of papers.
# 2. Do not forget to remove all the debug prints while submitting.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    # print(args)

    ################################################
    #               YOUR CODE START                #
    ################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the citation graph and papers
    
    G, papers = load_data()
    index_to_code = {paper['index']: paper['title'] for paper in papers}


    dataG = from_networkx(G)
    dataG = dataG.to(device)

    # perform preprocessing
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    tokmodel = AutoModel.from_pretrained("allenai/longformer-base-4096").to(device)
    tokmodel.eval()

    paper_embeddings = embed_all_papers(papers, tokenizer, tokmodel, device)
    # print(len(paper_embeddings))
    dataG.x = torch.stack([paper_embeddings[i] for i in range(dataG.num_nodes)])


    hidden_dim = next(iter(paper_embeddings.values())).shape[0]
    # print("debug 0")
    if (train_override==True) or (not (os.path.exists('data/graphsage_model.pth'))):
        train_GraphSAGE(dataG, paper_embeddings, device)

    with open('data/graphsage_embeddings.pkl', 'rb') as f:
        graphsage_embeddings = pickle.load(f)
    # print("debug 1")
    
    if (train_override == True) or (not (os.path.exists('data/projection_mlp.pth'))):
        train_projection_mlp(paper_embeddings, graphsage_embeddings, device)
    # print("debug 2")

    if (train_override == True) or (not os.path.exists('data/link_predictor.pth')):

        citation_edges = [
            (int(edge[0]), int(edge[1])) for edge in dataG.edge_index.cpu().numpy().T.tolist()
        ]

        train_link_predictor(graphsage_embeddings, citation_edges, device)
    # print("debug 3")
    
    projection_mlp = ProjectionMLP().to(device)
    projection_mlp.load_state_dict(torch.load("data/projection_mlp.pth", map_location=device))
    projection_mlp.eval()

    link_predictor = LinkPredictor().to(device)
    link_predictor.load_state_dict(torch.load("data/link_predictor.pth", map_location=device))
    link_predictor.eval()
    # print("debug 4")
    
    ranked = rank_papers(args.test_paper_title, args.test_paper_abstract, projection_mlp, tokenizer, tokmodel, link_predictor, device)
    ranked_codes = [index_to_code[i] for i in ranked]

    # prepare a ranked list of papers like this:
    result = ranked_codes  # Replace with your actual ranked list


    ################################################
    #               YOUR CODE END                  #
    ################################################


    
    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()