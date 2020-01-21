# Siamese Network (using tensorflow) on Quora duplication questions problem
Text Siamese Network provides a CNN based implementation of Siamese Network to solve Quora duplicate questions identification problem.
Quora question pair dataset has ~400k question pairs along with a binary label which states whether a pair of questions are similar or dissimilar. The Siamese Network based tries to capture the semantic similarity between questions.

## Requirements
- Python 3
- Pip 3
- Tensorflow
- FastText
- faiss

## Environment Setup
Execute requirements.txt to install dependency packages
```bash
pip install -r requirements.txt
```

## Training
1. Quora questions dataset is provided in ./data_repository directory. 
2. To train 
```bash
python train_siamese_network.py
```
## Prediction
Open Prediction.ipynb using Jupyter Notebook to look into Prediction module.

## Results
Given Question: **"Is it healthy to eat egg whites every day?"** most similar questions are as follows:
1. is it bad for health to eat eggs every day
2. is it healthy to eat once a day
3. is it unhealthy to eat bananas every day
4. is it healthy to eat bread every day
5. is it healthy to eat fish every day
6. what high protein foods are good for breakfast
7. how do you drink more water every day
8. what will happen if i drink a gallon of milk every day
9. is it healthy to eat one chicken every day
10. is it healthy to eat a whole avocado every day

Due to limitation in max file size in git, I haven't uploaded trained model in git. You can download pre-trained model from [here](https://drive.google.com/drive/folders/1FEdvcQt-tbNCZeUKhawFxyAn6Dn7H08I?usp=sharing) and unzip and paste pre-trained model to "./model_siamese_network" directory.

## Note
To train on a different dataset, you have to build a dataset consisting of similar and dissimilar text pairs. Empirically, you need to have at least ~200k number of pairs to achieve excellent performance. Try to maintain a balance between similar and dissimilar pairs [50% - 50%] is a good choice. 






