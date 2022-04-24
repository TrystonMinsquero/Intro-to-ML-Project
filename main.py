import streamlit as st
import datetime

def load_tsv(file):
    lines = file.readlines()

    assert len(lines) > 0, "file cannot be empty"

    # Get columns
    columns = [column.strip().strip("\"") for column in lines[0].split("\t")]

    def parse_line(line):
        split = line.split("\t")
        assert len(split) == len(columns), f"expected {len(columns)} columns, got {len(split)}"
        return dict([(columns[i], value.strip().strip("\"")) for i, value in enumerate(split)])

    return [parse_line(line) for line in lines[1:]]

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

f = open("amazon_alexa.tsv", "r", encoding="utf-8")
values = parse_reviews(load_tsv(f))

st.write(values)
