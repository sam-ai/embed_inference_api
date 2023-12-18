import argparse
import asyncio
import functools
import json
import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI, BackgroundTasks, File, Body, UploadFile, Request
from fastapi.responses import StreamingResponse
# from faster_whisper import WhisperModel
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
# from zhconv import convert

# from utils.data_utils import remove_punctuation
# from utils.utils import add_arguments, print_arguments


import hashlib
import os
import tarfile
import urllib.request

# from tqdm import tqdm


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
    model = SentenceTransformer(args.model_path, device="cuda", compute_type="float16")
else:
    model = SentenceTransformer(args.model_path, device='cpu')


# 
# _, _ = model.transcribe("dataset/test.wav", beam_size=5)

app = FastAPI(title="embedding Inference")
# app.mount('/static', StaticFiles(directory='static'), name='static')
# templates = Jinja2Templates(directory="templates")
# model_semaphore = None


# def release_model_semaphore():
#     model_semaphore.release()


# def recognition(file: File, to_simple: int,
#                 remove_pun: int, language: str = "bn",
#                 task: str = "transcribe"
#     ):

#     segments, info = model.transcribe(file, beam_size=10, task=task, language=language, vad_filter=args.vad_filter)
#     for segment in segments:
#         text = segment.text
#         if to_simple == 1:
#             # text = convert(text, '')
#             pass
#         if remove_pun == 1:
#             # text = remove_punctuation(text)
#             pass
#         ret = {"result": text, "start": round(segment.start, 2), "end": round(segment.end, 2)}
#         # 
#         yield json.dumps(ret).encode() + b"\0"


# @app.post("/recognition_stream")
# async def api_recognition_stream(
#         to_simple: int = Body(1, description="", embed=True),
#         remove_pun: int = Body(0, description="", embed=True),
#         language: str = Body("bn", description="", embed=True),
#         task: str = Body("transcribe", description="", embed=True),
#         audio: UploadFile = File(..., description="")
#         ):

#     global model_semaphore
#     if language == "None": language = None
#     if model_semaphore is None:
#         model_semaphore = asyncio.Semaphore(5)
#     await model_semaphore.acquire()
#     contents = await audio.read()
#     data = BytesIO(contents)
#     generator = recognition(
#         file=data, to_simple=to_simple,
#         remove_pun=remove_pun, language=language,
#         task=task
#         )
#     background_tasks = BackgroundTasks()
#     background_tasks.add_task(release_model_semaphore)
#     return StreamingResponse(generator, background=background_tasks)


@app.post("/embed")
async def api_embed(
            textA: str = Body("text1", description="", embed=True),
            textB: str = Body("text2", description="", embed=True),
        ):

    q_embeddings = model.encode(textA, normalize_embeddings=True)
    p_embeddings = model.encode(textB, normalize_embeddings=True)

    scores = q_embeddings @ p_embeddings.T
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