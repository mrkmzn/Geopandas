import pandas as pd
import csv
from mlxtend.frequent_patterns import apriori, association_rules

# make a dictionary to make a dataframe later
data_dict = {}

# read from csv of baskets
with open('groceries (2 prob).csv', 'r') as f:
    reader = csv.reader(f) 
    for index, row in enumerate(reader):
        data_dict[index] = row 
# make dataframe
products = pd.DataFrame.from_dict(data_dict, orient='index')

# convert dataframe in binary matrix
# decompose every basket into sets of products
basket = products.apply(lambda x: pd.Series(x.dropna().unique()), axis=1).stack().reset_index(level=1, drop=True)
basket.name = 'product'  # Set the name for the series
# each row as basket and columns are every product with binary identifier
basket_df = basket.reset_index().groupby(['index', 'product'])['product'].count().unstack().reset_index().fillna(0)
# Convert column names to strings
basket_df.columns = basket_df.columns.astype(str)
# set the transaction index
basket_df = basket_df.set_index('index').astype(bool) #astype.book requested by python


# apriori algorithm
frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)

# association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
