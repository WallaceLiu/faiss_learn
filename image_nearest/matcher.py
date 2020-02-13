from fsearcher import *


fs = FaissSearch()
result = fs.search_by_image('search/avena.JPG')
print(result)