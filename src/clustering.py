import logging
import configparser

from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from consts import COLUMNS
from loader import DataLoader

INPUT_FILE = "./experiments/market_data.parquet"
OUTPUT_FILE = "./experiments/clusters.parquet"
N_CLUSTERS = 25
NUM_PARTITIONS = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = "./config/config.ini"
config = configparser.ConfigParser()
config.read(config_path)
loader = DataLoader(config)

config = SparkConf()
sc = SparkContext(conf=config)
spark = SparkSession(sc)

df_datamart = spark.read.parquet(INPUT_FILE).repartition(NUM_PARTITIONS)
assemble = VectorAssembler(inputCols=COLUMNS, outputCol='features')
assembled_data = assemble.transform(df_datamart)

evaluator = ClusteringEvaluator(
    featuresCol='standardized',
    metricName='silhouette',
    distanceMeasure='squaredEuclidean'
)
scale = StandardScaler(
    inputCol='features',
    outputCol='standardized'
)

kmeans_model = KMeans(
    k=N_CLUSTERS,
    seed=42,
    featuresCol='standardized',
    predictionCol='prediction'
)

data_scale = scale.fit(assembled_data).transform(assembled_data)
model = kmeans_model.fit(data_scale)
transform_data = model.transform(data_scale)

evaluation_score = evaluator.evaluate(transform_data)
logger.info(f"EVALUATION SCORE: {evaluation_score}")

df_datamart_pd = df_datamart.toPandas()
df_datamart_pd["cluster"] = transform_data.select("prediction").toPandas().values.flatten()
loader.upload_prediction(df_datamart_pd)
