import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("netflix_titles.csv")

#data cleaning process:

df = df.loc[:, ~df.columns.str.contains("^Unnamed")] 

df.fillna({
    "director": "Unknown",
    "cast": "Unknown",
    "country": "Unknown",
    "date_added": "Unknown",
    "rating": "Unknown",
    "duration": "Unknown"}, inplace=True) 

df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year

df.to_csv("netflix_titles_cleaned.csv", index=False)

#Exploratory Data Analysis:

print(df.info()) 

#data visualizations:

# 1) movies VS Tvshows? 

type_counts = df["type"].value_counts()
plt.pie(type_counts, labels=type_counts.index, colors=["#6698FF","#87CEFA"])
plt.title("Distribution of Movies vs. TV Shows")
plt.legend()
plt.show()

# 2) Top 10 Countries Producing Content?

df["country"].value_counts().head(10).plot(kind="bar", color="#00BFFF")
plt.title("Top 10 Countries Producing HIGH Contents")
plt.xlabel("Country")
plt.ylabel("Count")
plt.legend()
plt.show()

# 3) Most Common Genres?

df["listed_in"].str.split(", ").explode().value_counts().head(10).plot(kind="line", color="#14A3C7", marker="o", ms=10, mfc="#6698FF", ls="--") 
plt.title("Top 10 Most Common Genres")
plt.show()

# 4) Most Popular Directors?

df["director"].value_counts().head(10).plot(kind="barh", color="#728FCE") 
plt.title("Top 10 Most Popular Directors")
plt.xlabel("Number of Movies/TV Shows")
plt.ylabel("Director")
plt.show()

# 5) Which year has the most films?

sns.kdeplot(df["release_year"], color="#14A3C7", fill=True, linewidth=2) 
plt.title("Distribution of Movies by Release Year (KDE)")
plt.xlabel("Release Year")
plt.ylabel("Density")
plt.show() 

# 6) Count of Content Added Per Year by "date_added" col?

df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month
heatmap_data = df.pivot_table(index="month_added", columns="year_added", aggfunc="size", fill_value=0)
sns.heatmap(heatmap_data, cmap="Blues", linewidths=0.5, annot=True, fmt="d")
plt.title("Netflix Content Added Per Year (by Month)")
plt.xlabel("Year Added")
plt.ylabel("Month Added")
plt.show() 

# 7) Top 10 Actors Appearing in Most Movies/Shows?

actors = df["cast"].str.split(", ")
actors = actors.explode()
actors = actors.value_counts()
top10_actors = actors.head(10)
top10_actors.plot(kind="bar", color= "#123456")
plt.title("TOP-10 Actors in NETFLIX")
plt.xlabel("Actor Names")
plt.ylabel("Count of Films/Shows")
plt.show()

#Regressions:

df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year

year_counting = df["year_added"].value_counts().sort_index()

X = np.array(year_counting.index).reshape(-1, 1)  # Years (Independent Variable)
y = np.array(year_counting.values)  # Content Count (Dependent Variable)

model = LinearRegression() 
model.fit(X, y) 

future_years = np.array([[2025], [2026], [2027]])
predictions = model.predict(future_years)
print("Predicted Content Addition for 2025,2036,2027:", predictions)

plt.scatter(X, y, color="#368BC1", label="Actual Data")
plt.plot(X, model.predict(X), color="#191970", label="Regression Line")
plt.xlabel("Year")
plt.ylabel("Content Added")
plt.legend()
plt.show()
