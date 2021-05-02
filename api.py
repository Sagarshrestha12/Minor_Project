
import requests
import numpy as np
import json



url = 'http://127.0.0.1:5000/results'



json1={'age':5, 'sex':1, 'cp':400, 'trestbps':200, 'chol':211, 'fbs':2, 'restecg':10,'thalach':1,'exang':1,'oldpeak':200,'slope':2,'ca':1,'thal':1}
r = requests.post(url,json = json1)

print(r.json())