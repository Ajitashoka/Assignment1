import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

print("Customers Dataset:")
print(customers.head())

print("\nProducts Dataset:")
print(products.head())

print("\nTransactions Dataset:")
print(transactions.head())

print("\nMissing Values in Customers Dataset:")
print(customers.isnull().sum())

print("\nMissing Values in Products Dataset:")
print(products.isnull().sum())

print("\nMissing Values in Transactions Dataset:")
print(transactions.isnull().sum())

customers.fillna("Unknown", inplace=True)
products.fillna("Unknown", inplace=True)
transactions.fillna(0, inplace=True)

print("\nDuplicates in Customers Dataset:", customers.duplicated().sum())
print("Duplicates in Products Dataset:", products.duplicated().sum())
print("Duplicates in Transactions Dataset:", transactions.duplicated().sum())

customers.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)

customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

print("\nCustomer Region Distribution:")
print(customers['Region'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(data=customers, x='Region', order=customers['Region'].value_counts().index)
plt.title("Customer Region Distribution")
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

print("\nSignup Date Trends:")
signup_trends = customers['SignupDate'].dt.to_period('M').value_counts().sort_index()
signup_trends.plot(kind='line', figsize=(10, 6))
plt.title("Signup Trends Over Time")
plt.xlabel("Month-Year")
plt.ylabel("Number of Signups")
plt.show()

print("\nProduct Category Distribution:")
print(products['Category'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(data=products, x='Category', order=products['Category'].value_counts().index)
plt.title("Product Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

print("\nProduct Price Distribution:")
sns.histplot(products['Price'], kde=True, bins=20)
plt.title("Product Price Distribution")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.show()

print("\nTransaction Value Distribution:")
sns.histplot(transactions['TotalValue'], kde=True, bins=20)
plt.title("Transaction Value Distribution")
plt.xlabel("Total Transaction Value (USD)")
plt.ylabel("Frequency")
plt.show()

print("\nMonthly Sales Trends:")
monthly_sales = transactions.groupby(transactions['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
monthly_sales.plot(kind='line', figsize=(10, 6))
plt.title("Monthly Sales Trends")
plt.xlabel("Month-Year")
plt.ylabel("Total Sales (USD)")
plt.show()

customers.to_csv("Cleaned_Customers.csv", index=False)
products.to_csv("Cleaned_Products.csv", index=False)
transactions.to_csv("Cleaned_Transactions.csv", index=False)
