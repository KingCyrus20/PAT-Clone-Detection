import urllib.request
import tarfile
import os

def download_code2vec():
    print("code2vec model not found in ./code2vec/models/java14_model, downloading from https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz")
    filename = "java14m_model.tar.gz"
    urllib.request.urlretrieve("https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz", filename)
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall("./code2vec")
    os.remove(filename)
