# Instacart Customer Analysis
Data for this project could be found at [here](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)  
Please create a new folder named "data" and put all the raw data files in it.


## Project Overview 
* Exploratory data analysis of products, orders, and customers.
* Text analysis on product name. 
* Computed embeddings for users that bought a lot of produce (main products at Instacart).
* Customer segmentation based on historical orders using k-means.
* Personalized marketing insights from segmentation results.


## EDA
[EDA Notebook](https://github.com/Veronica73/instacart_customer_analysis/blob/main/EDA.ipynb)  

Here are some examples in the exploratory data analysis:
* distribution of total number of orders placed by customers
* total sales amount by department
* percentage of reordered items

## Customer Segmentation Modeling
[Customer Segmentation Notebook](https://github.com/Veronica73/instacart_customer_analysis/blob/main/user_segmentation.ipynb). 

For segmentation, I calculated the number of items purchased from each department, and scaled them to be percentage value. Each customer is represented by a vector of $R^n$, where $n$ is the number of departments. All the feature values sum up to 1.

I used silhoutte score and validity index methods to determine the number of clusters in the data, which is six in the notebook. Then I fitted **k-means** model and generated some nice interpretations for the computed clusters, along with customized marketing insights for each of them.

## Product Name Text Analysis 
[Product Name Analysis Notebook](https://github.com/Veronica73/instacart_customer_analysis/blob/main/product_name_text_analysis.ipynb) 

First, I extracted product name from historical order data, tokenized and lemmatized them. I also removed punctuations, numbers and other irrelevant marks. Then I created visualization using word cloud. 

Besides, I computed embeddings for users in the "produce enthusaist" segmentation using **TF-IDF** method. Here, the customers' personal order history is the "document", and the entire order data as "corpus". I calculated **TF-IDF** scores for each word in the "document", and obtained a vector representation for each customer. The embeddings could be used for further interesting explorations, and some ideas are given in the notebook.

Happy Analyzing!
