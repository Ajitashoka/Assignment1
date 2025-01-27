import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

customers = pd.read_csv("/Users/ajit/Downloads/Customers.csv")
transactions = pd.read_csv("/Users/ajit/Downloads/Transactions.csv")
products = pd.read_csv("/Users/ajit/Downloads/Products.csv")

transactions = transactions.merge(products, on="ProductID")
data = transactions.merge(customers, on="CustomerID")

customer_features = (
    data.groupby("CustomerID")
    .agg({
        "TotalValue": "sum",
        "ProductID": "nunique",
        "Region": "first",
        "SignupDate": "first",
    })
    .reset_index()
)
customer_features.columns = ["CustomerID", "TotalSpend", "UniqueProducts", "Region", "SignupDate"]

customer_features["SignupYear"] = pd.to_datetime(customer_features["SignupDate"]).dt.year
encoder = OneHotEncoder(sparse=False)
region_encoded = encoder.fit_transform(customer_features[["Region"]])

scaler = StandardScaler()
numerical_data = scaler.fit_transform(customer_features[["TotalSpend", "UniqueProducts", "SignupYear"]])

X = torch.tensor(
    torch.cat((numerical_data, torch.tensor(region_encoded, dtype=torch.float32)), dim=1),
    dtype=torch.float32,
)

def create_pairs(features, customer_ids):
    pairs = []
    labels = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append((features[i], features[j]))
            labels.append(1 if customer_ids[i] == customer_ids[j] else 0)
    return pairs, labels

customer_ids = customer_features["CustomerID"].tolist()
pairs, labels = create_pairs(X, customer_ids)
pairs_tensor = [(torch.tensor(a), torch.tensor(b)) for a, b in pairs]
labels_tensor = torch.tensor(labels, dtype=torch.float32)

train_pairs, test_pairs, train_labels, test_labels = train_test_split(
    pairs_tensor, labels_tensor, test_size=0.2, random_state=42
)

class SiameseNetwork(nn.Module):
    def __init__(self, input_size):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x1, x2):
        embedding1 = self.fc(x1)
        embedding2 = self.fc(x2)
        return embedding1, embedding2

    def calculate_distance(self, x1, x2):
        return torch.norm(x1 - x2, dim=1)

input_size = X.shape[1]
model = SiameseNetwork(input_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, pairs, labels, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for (x1, x2), y in zip(pairs, labels):
            optimizer.zero_grad()
            embedding1, embedding2 = model(x1, x2)
            distance = model.calculate_distance(embedding1, embedding2)
            loss = criterion(distance, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(pairs)}")

train_model(model, train_pairs, train_labels, criterion, optimizer)

def find_lookalikes(model, customer_features, customer_id, top_n=3):
    customer_index = customer_features[customer_features["CustomerID"] == customer_id].index[0]
    target_features = torch.tensor(customer_features.iloc[customer_index, 1:], dtype=torch.float32)
    distances = []
    for i, row in customer_features.iterrows():
        if i != customer_index:
            other_features = torch.tensor(row[1:], dtype=torch.float32)
            _, embedding1 = model(target_features, target_features)
            _, embedding2 = model(other_features, other_features)
            distance = model.calculate_distance(embedding1, embedding2)
            distances.append((row["CustomerID"], distance.item()))
    distances.sort(key=lambda x: x[1])
    return distances[:top_n]

lookalikes = {}
for customer_id in customer_features["CustomerID"][:20]:
    lookalikes[customer_id] = find_lookalikes(model, customer_features, customer_id)

lookalike_df = pd.DataFrame({
    "CustomerID": lookalikes.keys(),
    "Lookalikes": [str(val) for val in lookalikes.values()]
})
lookalike_df.to_csv("Advanced_Lookalike.csv", index=False)
