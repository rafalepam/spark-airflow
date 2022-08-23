import sys
from pyspark.sql import SparkSession, Window
from pyspark import SparkContext
# from pyspark.sql.functions import col, year, month, rank, desc, sum, max, round
from pyspark.sql import functions as F
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table

sc = SparkContext()

spark = SparkSession.builder.getOrCreate()

logFile = sys.argv[1]

logData = sc.textFile(logFile).cache()

path = "/usr/local/spark/resources/data"

customers_df = spark.read.option("inferSchema", "True").option("header", "true") \
                         .csv(path + "/olist_customers_dataset.csv")

orders_df = spark.read.option("inferSchema", "True").option("header", "true") \
                      .csv(path + "/olist_orders_dataset.csv")

payments_df = spark.read.option("inferSchema", "True").option("header", "true") \
                        .csv(path + "/olist_order_payments_dataset.csv")

df = orders_df.agg(F.max(F.year("order_approved_at")).alias("a"))
n = df.toPandas().to_string()
last_year = int(n[11:15]) - 1
window = Window.partitionBy("month").orderBy(F.desc("total"))
df10 = orders_df.join(payments_df, payments_df.order_id == orders_df.order_id).drop(payments_df.order_id) \
                .join(customers_df, customers_df.customer_id == orders_df.customer_id).drop(orders_df.customer_id) \
                .filter(F.year("order_approved_at") == last_year) \
                .groupBy(["customer_unique_id", F.month("order_approved_at").alias("month")]) \
                .agg(F.sum("payment_value").alias("total")).orderBy(F.desc("total")).withColumn("rank", F.rank().over(window)) \
                .filter(F.col('rank') == 1).select("customer_unique_id", "month", F.round(F.col("total"), 2).alias("total")).orderBy("month")

data = [int(i * 100) / 100 for i in df10.select("total").orderBy(F.desc("month")).rdd.flatMap(lambda x: x).collect()]
cust = df10.select("customer_unique_id").orderBy(F.desc("month")).rdd.flatMap(lambda x: x).collect()
months = ['December', 'November', 'October', 'September',  'August', 'July',
          'June', 'May',  'April', 'March', 'February', 'January']

y_pos = np.arange(len(data))

plt.figure(figsize=(23, 10))

plt.barh(y_pos, data, color='gold')

plt.yticks(y_pos, months, va="center")

plt.xlabel('Monthly spending', fontsize=14, color='#323232')
plt.title('Top spending customers in ' + str(last_year), fontsize=16, color='#323232')

for i in range(len(cust)):
    plt.text(100, i, cust[i], color="black", fontsize=10)
    plt.text(data[i] + 50, i, data[i], color="black", fontsize=12)

# plt.show()
plt.savefig(path + "/img_1.png")

"""------------------------------------------------------------------------------------------------------------------"""

df11 = orders_df.join(payments_df, payments_df.order_id == orders_df.order_id).drop(payments_df.order_id) \
                .join(customers_df, customers_df.customer_id == orders_df.customer_id).drop(orders_df.customer_id) \
                .filter(F.year("order_approved_at") == last_year) \
                .groupBy([F.month("order_approved_at").alias("month")]) \
                .agg(F.sum("payment_value").alias("total")).orderBy(F.desc("total")).withColumn("rank", F.rank().over(window)) \
                .filter(F.col('rank') == 1).select("month", F.round("total", 2).alias("total")).orderBy("month")

data_2 = [int(i * 100) / 100 for i in df11.select("total").orderBy(F.desc("month")).rdd.flatMap(lambda x: x).collect()]

y_pos = np.arange(len(data_2))

plt.figure(figsize=(23, 10))

plt.barh(y_pos, data_2, color='deepskyblue')

plt.yticks(y_pos, months, va="center")

plt.xlabel('Monthly spending', fontsize=14, color='#323232')
plt.title('Total customers spending in ' + str(last_year), fontsize=16, color='#323232')

for i in range(len(data_2)):
    plt.text(1000 + data_2[i], i, round(data_2[i]), color="black", fontsize=12)

# plt.show()
plt.savefig(path + "/img_3.png")

"""------------------------------------------------------------------------------------------------------------------"""

window_2 = Window.partitionBy().orderBy(F.desc("total"))
df12 = orders_df.join(payments_df, payments_df.order_id == orders_df.order_id).drop(payments_df.order_id) \
                .join(customers_df, customers_df.customer_id == orders_df.customer_id).drop(orders_df.customer_id) \
                .filter(F.year("order_approved_at") == last_year) \
                .groupBy("customer_unique_id") \
                .agg(F.sum("payment_value").alias("total")).orderBy(F.desc("total")).withColumn("rank", F.rank().over(window_2)) \
                .filter(F.col("rank") <= 10).orderBy(F.desc("total")).select("rank", "customer_unique_id", F.round("total", 2).alias("total"))

data_3 = [int(i * 100) / 100 for i in df12.select("total").orderBy("total").rdd.flatMap(lambda x: x).collect()]
cust = df12.select("customer_unique_id").orderBy(F.desc("total")).rdd.flatMap(lambda x: x).collect()

y_pos = np.arange(len(data_3))

plt.figure(figsize=(23, 10))

plt.barh(y_pos, data_3, color='gold')

plt.yticks(y_pos, [i for i in range(10, 0, -1)], va="center")

plt.title('Ranking of Customer spending in ' + str(last_year), fontsize=16, color='#323232')

for i in range(len(data_3)):
    plt.text(100, i, cust[i], color="black", fontsize=10)
    plt.text(100 + data_3[i], i, round(data_3[i]), color="black", fontsize=12)

# plt.show()
plt.savefig(path + "/img_5.png")

"""------------------------------------------------------------------------------------------------------------------"""

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
# ax.set_frame_on(False)
table(ax, df10.toPandas(), loc='center')  # where df is your data frame
plt.savefig(path + '/img_2.png')

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
# ax.set_frame_on(False)
table(ax, df11.toPandas(), loc='center')  # where df is your data frame
plt.savefig(path + '/img_4.png')

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
# ax.set_frame_on(False)
table(ax, df12.toPandas(), loc='center')  # where df is your data frame
plt.savefig(path + '/img_6.png')


from fpdf import FPDF


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        # Custom logo and positioning
        # Create an `assets` folder and put any wide and short image inside
        # Name the image `logo.png`
        #         self.image('assets/logo.png', 10, 8, 33)
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, 'Costumer Spending report', 0, 0, 'R')
        self.ln(20)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_body(self, images):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        if len(images) == 3:
            self.image(images[0], 0, 25, self.WIDTH)
            self.image(images[1], 0, self.WIDTH / 2 + 5, self.WIDTH)
            self.image(images[2], 0, self.WIDTH / 2 + 90, self.WIDTH)
        elif len(images) == 2:
            self.image(images[0], 0, 25, self.WIDTH)
            self.image(images[1], 0, self.WIDTH / 2 + 5, self.WIDTH)
        else:
            self.image(images[0], 0, 25, self.WIDTH)

    def print_page(self, images):
        # Generates the report
        self.add_page()
        self.page_body(images)


def construct():
    counter = 0
    pages_data = []
    temp = []
    # Get all plots
    files = [image for image in os.listdir(path) if image.endswith('png')]
    # Sort them by month - a bit tricky because the file names are strings
    # Iterate over all created visualization
    for fname in files:
        # We want 3 per page
        if counter == 3:
            pages_data.append(temp)
            temp = []
            counter = 0

        temp.append(f'{path}/{fname}')
        counter += 1

    return [*pages_data, temp]


plots_per_page = construct()

pdf = PDF()

for elem in plots_per_page:
    pdf.print_page(elem)

pdf.output(path + '/CustomerSpendingReport.pdf', 'F')
