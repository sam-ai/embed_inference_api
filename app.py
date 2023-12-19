import argparse
import asyncio
import functools
import json
import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI, Body, Request
# from fastapi.responses import StreamingResponse
# from starlette.staticfiles import StaticFiles
# from starlette.templating import Jinja2Templates
from utils.utils import add_arguments, print_arguments
from sentence_transformers import SentenceTransformer, models

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)


add_arg("host",        type=str,  default="0.0.0.0", help="")
add_arg("port",        type=int,  default=5000,        help="")
add_arg("model_path",  type=str,  default="BAAI/bge-small-en-v1.5", help="")
add_arg("use_gpu",     type=bool, default=False,   help="")
add_arg("num_workers", type=int,  default=2,      help="")



args = parser.parse_args()
print_arguments(args)



# similarity score func
def similarity_score(model, textA, textB):
    em_test = model.encode(
        [textA, textB],
        normalize_embeddings=True
    )
    return em_test[0] @ em_test[1].T


# BGE embedding

if args.use_gpu:
    bge_model = SentenceTransformer(args.model_path, device="cuda", compute_type="float16", cache_folder=".")
else:
    bge_model = SentenceTransformer(args.model_path, device='cpu', cache_folder=".")


# tsdae embedding
if args.use_gpu:
    model_name = 'sam2ai/sbert-tsdae'
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    tsdae_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device="cuda",
        compute_type="float16",
        cache_folder="."
    )
else:
    model_name = 'sam2ai/sbert-tsdae'
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    tsdae_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device='cpu',
        cache_folder="."
    )


# word2vec embedding
# Define the calculate_similarity function
def calculate_similarity(sentence1, sentence2):
    # Tokenize the sentences
    tokens1 = simple_preprocess(sentence1)
    tokens2 = simple_preprocess(sentence2)

    # Load or train a Word2Vec model
    # Here, we'll create a simple model for demonstration purposes
    sentences = [tokens1, tokens2]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

    # Calculate the vector representation for each sentence
    vector1 = np.mean([model.wv[token] for token in tokens1], axis=0)
    vector2 = np.mean([model.wv[token] for token in tokens2], axis=0)

    # Calculate cosine similarity
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity



app = FastAPI(title="embedding Inference")

@app.get("/")
async def index(request: Request):
    return {"detail": "API is Active !!"}

@app.post("/bge_embed")
async def api_bge_embed(
            text1: str = Body("text1", description="", embed=True),
            text2: str = Body("text2", description="", embed=True),
        ):

    scores = similarity_score(bge_model, text1, text2)
    print(scores)
    scores = scores.tolist()

    ret = {"similarity score": scores, "status_code": 200}
    return ret

@app.post("/tsdae_embed")
async def api_tsdae_embed(
            text1: str = Body("text1", description="", embed=True),
            text2: str = Body("text2", description="", embed=True),
        ):

    scores = similarity_score(tsdae_model, text1, text2)
    print(scores)
    scores = scores.tolist()

    ret = {"similarity score": scores, "status_code": 200}
    return ret

@app.post("/w2v_embed")
async def api_w2v_embed(
            text1: str = Body("text1", description="", embed=True),
            text2: str = Body("text2", description="", embed=True),
        ):

    scores = calculate_similarity(text1, text2)
    print(scores)
    scores = scores.tolist()

    ret = {"similarity score": scores, "status_code": 200}
    return ret




if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port)