# Image nearest neighbors search API

This is an basic example on how to build an image search web service.

It uses [OpenCV](https://github.com/opencv/opencv) and [FAISS](https://github.com/facebookresearch/faiss) for image processing and indexing.

[FastAPI](https://github.com/tiangolo/fastapi) for the API endpoint.

### Configuration file

The general configuration for the indexer and for the search is in faissconfig.py


### Generating the index

To generate the index along with the pickled vector ids you will need to run the file indexer.py

```
ENV/bin/python indexer.py
iterating images from path.. images
training..
saving index..
```

once it's done you can now start the server

```
ENV/bin/uvicorn imagenearest.main:app --reload
```

### Make a request

The method `/search` accepts base64 encoded images. You can find an example in the file curl.txt

Executing a curl request to test the service

```
curl -H "Content-Type: application/json" --data @curl.txt http://127.0.0.1:8000/search
```
