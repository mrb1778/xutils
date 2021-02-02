import pyspark.sql.functions as f
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    DoubleType
)
from pyspark.sql import DataFrame


def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return f.udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)


def my_transformations(df_1: DataFrame) -> DataFrame:
    df_1 = df_1.select(
        'category',
        'main_category',
        'currency',
        f.col('goal').cast('float'),
        'launched',
        'deadline',
        f.col('pledged').cast('float'),
        'state',
        f.col('backers').cast('float')
    ).withColumn(
        'launched_unix', f.unix_timestamp('launched')
    ).withColumn(
        'deadline_unix', f.unix_timestamp('deadline', 'yyyy-MM-dd')
    ).withColumn(
        'time_span', f.col('deadline_unix').cast('long') -
                     f.col('launched_unix').cast('long')
    ).filter(
        (f.col('currency') == 'USD')
    ).filter(
        (f.col('state') == "failed") | (f.col('state') == "successful")
    ).filter(
        (f.col('goal') != 0.0)
    ).drop(
        'launched',
        'deadline',
        'launched_unix',
        'deadline_unix',
        'currency',
        'id'
    )

    return df_1