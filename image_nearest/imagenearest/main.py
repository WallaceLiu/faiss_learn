import json
import asyncio

from fastapi import Depends, FastAPI, HTTPException

from pydantic import BaseModel
import base64
import binascii
import numpy as np
import cv2 as cv
from fsearcher import *
import json


class ImageSearch(BaseModel):
    image: str
    vectors: str
    topN: int


app = FastAPI()


@app.get("/")
def _root():
    return {"msg": "Image nearest search"}


@app.post("/search")
def _search(image: ImageSearch):
    image = image.dict()
    if image.get("image") != "":
        img = image.get("image")
        topN = image.get("topN")
        fs = FaissSearch()
        result = fs.search_by_image(img)
        highest_match = max(result, key=lambda x: x["score"])
        similar = [s for s in result if s.get("id") != highest_match.get("id")]
        response = {"similar": similar, "match": highest_match}
        return response
    else:
        return {"status": "error", "msg": "invalid image"}
