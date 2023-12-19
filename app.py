import argparse
import asyncio
import functools
import json
import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI, BackgroundTasks, File, Body, UploadFile, Request
from fastapi.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer

# from utils.data_utils import remove_punctuation
# from utils.utils import add_arguments, print_arguments


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def str_none(val):
    if val == 'None':
        return None
    else:
        return val


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    type = str_none if type == str else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg("host",        type=str,  default="0.0.0.0", help="")
add_arg("port",        type=int,  default=5000,        help="")
add_arg("model_path",  type=str,  default="BAAI/bge-small-en-v1.5", help="")
add_arg("use_gpu",     type=bool, default=False,   help="")
# add_arg("use_int8",    type=bool, default=True,  help="")
add_arg("beam_size",   type=int,  default=10,     help="")
add_arg("num_workers", type=int,  default=2,      help="")
add_arg("vad_filter",  type=bool, default=True,  help="")
add_arg("local_files_only", type=bool, default=True, help="")
args = parser.parse_args()
print_arguments(args)

# 
# assert os.path.exists(args.model_path), f"{args.model_path}"
# 
if args.use_gpu:
    model = SentenceTransformer(args.model_path, device="cuda", compute_type="float16", cache_folder=".")
else:
    model = SentenceTransformer(args.model_path, device='cpu', cache_folder=".")


app = FastAPI(title="embedding Inference")
# app.mount('/static', StaticFiles(directory='static'), name='static')
# templates = Jinja2Templates(directory="templates")
# model_semaphore = None

def similarity_score(textA, textB):
    em_test = model.encode(
        [textA, textB],
        normalize_embeddings=True
    )
    return em_test[0] @ em_test[1].T


@app.post("/embed")
async def api_embed(
            text1: str = Body("text1", description="", embed=True),
            text2: str = Body("text2", description="", embed=True),
        ):

    scores = similarity_score(text1, text2)
    print(scores)
    scores = scores.tolist()

    ret = {"similarity score": scores, "status_code": 200}
    return ret


# @app.get("/")
# async def index(request: Request):
#     return templates.TemplateResponse(
#         "index.html", {"request": request, "id": id}
#         )


if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port)