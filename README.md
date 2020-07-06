# faiss_learn
faiss_learn

### 4.2 Local PySpark
- step 1：先安装指定py包
```shell script
pip install sklearn
pip install faiss-cpu
pip install jieba
#pip install smart_open
```
- step 2：再打包成zip
```shell script
cd /Users/liuning11/.conda/envs/myfaiss
zip -r myfaiss.zip *
```
- step 3：让pyspark能够使用py包
```shell script
pyspark --archives /Users/liuning11/.conda/envs/myfaiss/myfaiss.zip --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/Users/liuning11/.conda/envs/myfaiss/myfaiss.zip/bin/python
```
> 备注:
> ```
> --archives <环境包地址，最好放在hdfs集群上，免得上传>
> --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./XX.zip/XX/bin/python
> ```
> 如果是yarn-cluster模式，最好也设置下以下参数：
> ```
> spark.yarn.appMasterEnv.PYSPARK_PYTHON = ./XX.zip/XX/bin/python
> spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON = ./XX.zip/XX/bin/python
> ```

# 参考
- [官网文档](https://github.com/facebookresearch/faiss/wiki)
- [Faiss-indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [similarity search](https://github.com/facebookresearch/faiss/wiki)
- [Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki)
- [Benchmarking scripts](https://github.com/facebookresearch/faiss/blob/master/benchs/README.md)

# 欢迎打赏，sponsors.jpeg
