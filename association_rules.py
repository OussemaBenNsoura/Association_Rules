import pandas as pd
import matplotlib.pyplot as plt

import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori , association_rules


toy_dataset = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],
               ['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],
               ['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],
               ['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],
               ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]

te = TransactionEncoder()
te_train = te.fit(toy_dataset).transform(toy_dataset)
df = pd.DataFrame(te_train,columns=te.columns_)
print(df)
## implement the apriori concept
apriori_algo = apriori(df,min_support=0.6,use_colnames=True)
print(apriori_algo)
    # implement the association rule (confidence)
assosciation_df = pd.DataFrame(association_rules(apriori_algo, metric='confidence',min_threshold=0.5))
print(assosciation_df.info())
print(assosciation_df.iloc[:,:7])
    # implement the association rule (lift)
assosciation_df_lift = pd.DataFrame(association_rules(apriori_algo, metric='lift',min_threshold=1.0))
print(assosciation_df_lift.info())
print(assosciation_df_lift.iloc[:,:7])

data = pd.read_csv('Market_Basket_Optimisation.csv')
print(data.info())

# Compter les fréquences des articles
item_counts = data['avocado'].value_counts()


plt.figure(figsize=(10, 6))
plt.bar(item_counts.index, item_counts.values, color='skyblue')
plt.title('Fréquence des achats des articles')
plt.xlabel('Articles')
plt.ylabel('Fréquence')
plt.show()
