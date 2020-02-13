from fastapi import FastAPI
from starlette.testclient import TestClient
from imagenearest.main import app



client = TestClient(app)

def test_search():
    with open("curl.txt", 'rb') as jsonpayload:
        data = jsonpayload.read()
    response = client.post('/search', data)
    assert response.json() == {'id': 418, 'file_path': 'images/maclap.jpg', 'score': '14.046121593291405'}
        
    
