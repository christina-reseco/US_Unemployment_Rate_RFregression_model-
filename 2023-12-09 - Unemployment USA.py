# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This analysis and predictive Random Forest Regression model focuses on Unemployment rate in USA over 2000-2022.
# MAGIC
# MAGIC Data Source:https://www.ers.usda.gov/data-products/county-level-data-sets/

# COMMAND ----------

# MAGIC %md Import Data into a Dataframe table

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/UnemploymentUSA2.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
#delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
unemployment_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .load(file_location)

display(unemployment_df)

# COMMAND ----------

# MAGIC %md Import Python Libraries

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md Prepare and Transform Data

# COMMAND ----------

#Convert spark dataframe to pandas dataframe
unemployment_pdf = unemployment_df.select("*").toPandas()

# COMMAND ----------

unemployment_pdf.head()

# COMMAND ----------

unemployment_pdf.info()

# COMMAND ----------

#Select relevant columns from the original unemployment dataset
unemployment_rate = unemployment_pdf[['FIPS_Code', 'State', 'Area_Name', 'Rural_Urban_Continuum_Code_2013', 'Urban_Influence_Code_2013', 'Metro_2013', 'Civilian_labor_force_2000', 'Median_Household_Income_2021','Med_HH_Income_Percent_of_State_Total_2021', 'Unemployment_rate_2000','Unemployment_rate_2001','Unemployment_rate_2002','Unemployment_rate_2003','Unemployment_rate_2004','Unemployment_rate_2005','Unemployment_rate_2006','Unemployment_rate_2007','Unemployment_rate_2008','Unemployment_rate_2009','Unemployment_rate_2010', 'Unemployment_rate_2011', 'Unemployment_rate_2012', 'Unemployment_rate_2013','Unemployment_rate_2014','Unemployment_rate_2015','Unemployment_rate_2016','Unemployment_rate_2017','Unemployment_rate_2018','Unemployment_rate_2019','Unemployment_rate_2020','Unemployment_rate_2021','Unemployment_rate_2022']]

# COMMAND ----------

unemployment_rate.columns=['FIPS_Code', 'State', 'Area_Name', 'Rural_Urban_Continuum_Code_2013', 'Urban_Influence_Code_2013', 'Metro_2013', 'Civilian_labor_force_2000', 
 'Median_Household_Income_2021','Med_HH_Income_Percent_of_State_Total_2021', '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010', '2011', '2012', '2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']

# COMMAND ----------

unemployment_rate.info()

# COMMAND ----------

#Transform/unpivot dataframe to convert years into a column and populate unemployment rate across the years
melted = unemployment_rate.melt(id_vars=['FIPS_Code', 'State', 'Area_Name', 'Rural_Urban_Continuum_Code_2013', 'Urban_Influence_Code_2013', 'Metro_2013', 'Civilian_labor_force_2000', 'Median_Household_Income_2021', 'Med_HH_Income_Percent_of_State_Total_2021'], var_name='year', value_name='Unemployment_rate')

# COMMAND ----------

melted.info()

# COMMAND ----------

# MAGIC %md Data Discovery and Visualization
# MAGIC
# MAGIC - Yearly Trend in Unemployment Rate over time
# MAGIC - Yearly Trend in Unemployment Rate in Metro (Urban) and Non-Metro Communities

# COMMAND ----------

#Convert columns to integer data type 
melted['year']=melted['year'].astype(int)
melted['Unemployment_rate']=melted['Unemployment_rate'].astype(float)

# COMMAND ----------

melted['Median_Household_Income_2021']=melted['Median_Household_Income_2021'].astype(float)
melted['Med_HH_Income_Percent_of_State_Total_2021']=melted['Med_HH_Income_Percent_of_State_Total_2021'].astype(float)

# COMMAND ----------

melted['Median_Household_Income_2021']

# COMMAND ----------

#Group Unemployment Rate by Year to get historical trends
unemployment_yearly=pd.DataFrame(melted.groupby('year')['Unemployment_rate'].mean())
unemployment_yearly

# COMMAND ----------

unemployment_yearly.info()

# COMMAND ----------

unemployment_yearly.index

# COMMAND ----------

unemployment_yearly['year']=unemployment_yearly.index

# COMMAND ----------

unemployment_yearly

# COMMAND ----------

# MAGIC %md Yearly Trend in Unemployment Rate over time

# COMMAND ----------

#Plot Mean Unemployment Rate in USA by Year
plt.figure(figsize=(20, 8))
sns.barplot(x='year', y='Unemployment_rate', data=unemployment_yearly, palette='Blues')
plt.show()

# COMMAND ----------

#Group Unemployment rate data by Metro designation for each community
unemployment_metro=pd.DataFrame(melted.groupby(['year', 'Metro_2013'], as_index=False)['Unemployment_rate'].mean())
unemployment_metro

# COMMAND ----------

# MAGIC %md Yearly Trend in Unemployment Rate in Metro (Urban) and Non-Metro Communities
# MAGIC
# MAGIC During periods of relatively high unemployment, Metro locations have higher unemployment than non-metro.
# MAGIC
# MAGIC During periods of relatively lower unemployment, Metro and Non-Metro locations have more similar rates. 
# MAGIC
# MAGIC Urban jobs in Metro areas are more affected by rising unemployment.

# COMMAND ----------

#Plot Unemployment Rate by year and by Metro type of community (metro/urban and non-metro/urban)
plt.figure(figsize=(20,8))
sns.barplot(x='year', y='Unemployment_rate', data=unemployment_metro, hue='Metro_2013')
plt.show()

# COMMAND ----------

#Plot Median_Household_Income
plt.plot(figsize=(24, 6))
sns.set(rc={'figure.figsize':(20,8)})
from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 20,8

sns.set(font_scale=2)

sns.scatterplot(x='Median_Household_Income_2021', y='Unemployment_rate', data=melted, palette="crest")
plt.xlabel('Rural Urban Continuum Code')
plt.ylabel('Unemployment Rate')
plt.show()

# COMMAND ----------

unemployment_state=pd.DataFrame(melted.groupby('State')['Unemployment_rate'].mean().reset_index())

# COMMAND ----------

unemployment_statesorted = unemployment_state.sort_values(by=['Unemployment_rate'], ascending=False)

# COMMAND ----------

#Plot Mean Unemployment by State
plt.plot(figsize=(24, 6))
sns.set(rc={'figure.figsize':(20,8)})
from matplotlib import rcParams
sns.set(font_scale=1.2)

# figure size in inches
rcParams['figure.figsize'] = 20,8
sns.barplot(x='State', y='Unemployment_rate', data=unemployment_statesorted, palette="crest")
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md Prepare Data for Modeling

# COMMAND ----------

#Copy dataframe for subsequent data preparation for modeling
unemployment=melted.copy()

# COMMAND ----------

#Check Missing Values
unemployment.isnull().sum()

# COMMAND ----------

unemployment=unemployment.dropna()
unemployment.isnull().sum()

# COMMAND ----------

#Check data types of each column
unemployment.info()

# COMMAND ----------

unemployment.head()

# COMMAND ----------

unemployment['FIPS_Code']=unemployment['FIPS_Code'].astype(int)


# COMMAND ----------

unemployment['Rural_Urban_Continuum_Code_2013']=unemployment['Rural_Urban_Continuum_Code_2013'].astype('int')
unemployment['Urban_Influence_Code_2013']=unemployment['Urban_Influence_Code_2013'].astype(int)
unemployment['Metro_2013']=unemployment['Metro_2013'].astype(int)

unemployment['Median_Household_Income_2021']=unemployment['Median_Household_Income_2021'].astype(int)
unemployment['Med_HH_Income_Percent_of_State_Total_2021']=unemployment['Med_HH_Income_Percent_of_State_Total_2021'].astype(float)
unemployment.info()

# COMMAND ----------

#unemployment['Civilian_labor_force_2000']=unemployment['Civilian_labor_force_2000'].astype(int)
unemployment['Civilian_labor_force_2000']=unemployment['Civilian_labor_force_2000'].astype(str).astype(float)

# COMMAND ----------

unemployment.info()

# COMMAND ----------

unemployment.info()

# COMMAND ----------

#Plot Correlation Heatmap of Unemployment Features
plt.figure(figsize=(20,10))
# plotting correlation heatmap
sns.set(font_scale=2)

dataplot = sns.heatmap(unemployment.corr(), annot=True).set(title='Correlation Plot of US Unemployment Features')

# displaying heatmap
plt.show();

# COMMAND ----------

#Convert pandas dataframe to pandas-on-Spark DataFrame to proceed with modelling with MLlib
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("app").getOrCreate()
unemployment_pyspark_df = spark.createDataFrame(unemployment)

# COMMAND ----------

#Define feature columns (omit the index and target label column)
nonFeatureCols = ["Unemployment_rate"]
featureCols = ['FIPS_Code', 'Rural_Urban_Continuum_Code_2013', 'Urban_Influence_Code_2013', 'Metro_2013', 'Civilian_labor_force_2000', 'Median_Household_Income_2021', 'Med_HH_Income_Percent_of_State_Total_2021', 'year']

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
 
assembler = (VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features"))
 
finalunemployment = assembler.transform(unemployment_pyspark_df)

# COMMAND ----------

#Split data into train and test
training, test = finalunemployment.randomSplit([0.7, 0.3])
 
#  Cache the data 
training.cache()
test.cache()
 
print(training.count())
print(test.count())

# COMMAND ----------

#Random Forest Regression model with hyperparameter tuning
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
 
from pyspark.ml import Pipeline
 
rfModel = (RandomForestRegressor()
  .setLabelCol("Unemployment_rate")
  .setFeaturesCol("features"))
 
paramGrid = (ParamGridBuilder()
  .addGrid(rfModel.maxDepth, [5,10])
  .addGrid(rfModel.numTrees, [20,60])
  .build())

 
stages = [rfModel]
 
pipeline = Pipeline().setStages(stages)
 
cv = (CrossValidator() 
  .setEstimator(pipeline) 
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(RegressionEvaluator().setLabelCol("Unemployment_rate")))
 
pipelineFitted = cv.fit(training)

# COMMAND ----------

#Identify the best model from hyperparamter tuning - best model has 60 trees and 9 features, we passed 10 features, so most features are significant in predicting the diamond price, we can check the feature importance of this model as well (further below)
print("The Best Parameters:\n--------------------")
print(pipelineFitted.bestModel.stages[0])
pipelineFitted.bestModel.stages[0].extractParamMap()

# COMMAND ----------

pipelineFitted.bestModel

# COMMAND ----------

#Make predictions from the test dataset using the best model
holdout2 = pipelineFitted.bestModel.transform(test)

# COMMAND ----------

display(holdout2)

# COMMAND ----------

predictions_pdf = holdout2.select("*").toPandas()

# COMMAND ----------

#Plot Mean Unemployment Rate and Model Prediction in USA by Year
plt.figure(figsize=(20, 8))
#f, ax = plt.subplots(1, 1)
sns.set(font_scale=2)

sns.lineplot(x='year', y='Unemployment_rate', data=predictions_pdf, label='Unemployment Rate (Actual)')
sns.lineplot(x='year', y='prediction', data=predictions_pdf, label='Unemployment Rate (Prediction)')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md The Unemployment Rate prediction using the Random Forest Regression with Spark MLlib provides a good overall prediction of the trends over time (2000-2020) with MAE = 1.09 and R2=0.68. 
# MAGIC
# MAGIC The discrepancies in the predictions are mainly due to peaks and valleys in unemployment when there are significant increases/decreases in unemployment that are different from the overall trends. 
# MAGIC
# MAGIC These changes in unemployment happen due to economic events that are not easily predictable (ex. market crashes around 2000, 2008).

# COMMAND ----------

#Evaluate model performance using RegressionMetrics
from pyspark.mllib.evaluation import RegressionMetrics
 
rm2 = RegressionMetrics(holdout2.select("prediction", "Unemployment_rate").rdd.map(lambda x:  (float(x[0]), float(x[1]))))
print("MSE: ", rm2.meanSquaredError)
print("MAE: ", rm2.meanAbsoluteError)
print("RMSE Squared: ", rm2.rootMeanSquaredError)
print("R Squared: ", rm2.r2)
print("Explained Variance: ", rm2.explainedVariance, "\n")

# COMMAND ----------

#Feature Importances
featureImportance = pipelineFitted.bestModel.stages[0].featureImportances.toArray()
featureNames = map(lambda s: s.name, finalunemployment.schema.fields)
featureImportanceMap = zip(featureImportance, featureNames)

# COMMAND ----------

importancesDf = spark.createDataFrame(sc.parallelize(featureImportanceMap).map(lambda r: [r[1], float(r[0])]))

# COMMAND ----------

importancesDf = importancesDf.withColumnRenamed("_1", "Feature").withColumnRenamed("_2", "Importance")

# COMMAND ----------

#display(importancesDf)
importancesDf.orderBy("Importance").show(truncate=False)

# COMMAND ----------

#Plot Mean Unemployment Rate in USA by Year
plt.figure(figsize=(20, 8))
sns.set(font_scale=2)

sns.scatterplot(x='Median_Household_Income_2021', y='Unemployment_rate', data=predictions_pdf)
plt.xlabel('Median Household Income')
plt.ylabel('Unemployment Rate')
plt.show()

# COMMAND ----------

#Plot Mean Unemployment Rate in USA by Year
plt.figure(figsize=(20, 8))

sns.barplot(x='year', y='Unemployment_rate', data=predictions_pdf, hue='Metro_2013')
plt.show()
