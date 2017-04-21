import sys
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

# set up contexts
conf = SparkConf()
conf.set("spark.master", "local[*]")
conf = conf.setAppName('Pysparksvm')
sc = SparkContext(conf=conf)
sql = SQLContext(sc)
stream = StreamingContext(sc, 1)

topic = "my_data"
kafka_stream = KafkaUtils.createStream(stream, "localhost:2181", "topic", {topic: 1})
lines = kafka_stream.map(lambda x: x[1])

print(lines)

stream.start()
stream.awaitTermination()
