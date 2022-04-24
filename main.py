import streamlit as st
import datetime

# Loads a TSV file into an array of dictionaries
# The keys of the dictionary are the columns defined in the TSV
# and the values are what are in the following rows, naturally
# file is an actual file object returned by the open function
def load_tsv(file):
    lines = file.readlines()

    assert len(lines) > 0, "file cannot be empty"

    # Get columns
    columns = [column.strip().strip("\"") for column in lines[0].split("\t")]

    # Parses a single line of the TSV
    # Basically creates a dictionary of values using the column names retrived above
    def parse_line(line):
        split = line.split("\t")
        assert len(split) == len(columns), f"expected {len(columns)} columns, got {len(split)}"
        return dict([(columns[i], value.strip().strip("\"")) for i, value in enumerate(split)])

    # Parse the rest of the lines after the first one containing the columns
    return [parse_line(line) for line in lines[1:]]

# Takes a loaded TSV (from load_tsv) and converts everything to more easily usable data
# These functions are separate in case we need to parse anything else
def parse_reviews(values):
    def parse_review(value):
        result = {}
        result["rating"] = int(value["rating"])
        result["date"] = datetime.datetime.strptime(value["date"], "%d-%b-%y")
        result["variation"] = value["variation"]
        result["content"] = value["verified_reviews"]
        result["feedback"] = bool(value["feedback"])
        return result
    
    return [parse_review(value) for value in values]

# And this just tests out the code above
f = open("amazon_alexa.tsv", "r", encoding="utf-8")
values = parse_reviews(load_tsv(f))
st.write(values)
