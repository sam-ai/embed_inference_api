<h1 align ="center">embed_inference_api</h1>


## Supporting models
- BAAI/bge-small-en-v1.5
- sentence-transformers/all-MiniLM-L6-v2
- sam2ai/sbert-tsdae


## Introduction of the main program of the project

1. `.github\workflow`: script github Actions Continuous deployment to hugginface hub space.
2. `notebooks`: experiments with different models.
3. `utils`: scripts for server.
4. `app.py`: Main server code.
5. `Dockerfile`: docker script to build container.
6. `download.py`: script to download models to a predefined folder.
7. `requirement.txt`: all require libraries to build an app.

8. ## deploy

deployment can be expedited using hugginface space server using docker. 
- `--host` option designates the address at which the service will initiate, denoted as `0.0.0.0`  - `--port` parameter specifies the port number for usage.
- `--model_path` parameter designates the bge model path.
- `--num_workers` parameter specifies the number of threads for concurrent inference


run by Python env setup

```shell
pip install -r requirements.txt

python infer_server.py --host=0.0.0.0 --port=5000 --model_path=BAAI/bge-small-en-v1.5 --num_workers=2
```

run by Docker env setup

```shell
docker build -t embed-api:latest .
docker run -p 7860:7860 embed-api:latest
```

### API docs

Currently, there are three available inferences:
- `/bge_embed` (POST) BGE-embedding-based API
- `/w2v_embed` (POST) word2vector based API
- `/tsdae_embed` (POST) transformer-based denoising autoencoder API

  swagger API docs URL: https://sam2ai-embed-api.hf.space/docs

|   Field    | Need |  type  |  Default   |                                  Explain                                  |
|:----------:|:----:|:------:|:----------:|:-------------------------------------------------------------------------:|
|   text1    | Yes  |  str   |            |                                text1 long paragraph                       |
|   text2    | Yes  |  str   |            |                                text2 long paragraph                       |

Example:

```json
{
  "text1": "......................",
  "text2": "......................."
}
```

Return result:

|  Field  | type |                       Explain                       |
|:-------:|:----:|:---------------------------------------------------:|
| similarity score | float  | similarity score between text1 & text2   |
| status_code | int  |                      |

Example:

```json
{
    "similarity score": 0.7884424924850464,
    "status_code": 200
}
```

Python request
```python
import requests
import json

url = "https://sam2ai-embed-api.hf.space/tsdae_embed"

payload = json.dumps({
  "text1": "...................",
  "text2": "....................."
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

```

Hugginface space URL : https://sam2ai-embed-api.hf.space
