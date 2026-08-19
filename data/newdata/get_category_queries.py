import json


with open("search_queries.json") as fin:
    data = json.load(fin)

cat_to_query = {}
for datapiece in data["categories"]:
    cat_to_query[datapiece["category_name"]] = []
    for query in datapiece["queries"]:
        cat_to_query[datapiece["category_name"]].append(query["query_text"])

with open("category_queries.json", "w") as fout:
    json.dump(cat_to_query, fout, indent=4)