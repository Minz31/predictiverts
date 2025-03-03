from pathlib import Path
from typing import List, Tuple
import re

# Attempt to import scikit-learn components.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as e:
    raise ImportError(
        "scikit-learn is required for TFIDFbaseline. Please install scikit-learn (version >= 1.2) "
        "which is compatible with Python 3.13."
    ) from e

# Regular expression for subtokenization
RE_SUBTOKENIZE = re.compile(
    r"(?<=[_$])(?!$)|(?<!^)(?=[_$])|(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[0-9]|[A-Z][a-z0-9])|(?<=[0-9])(?=[a-zA-Z])"
)


def is_identifier(token: str) -> bool:
    return len(token) > 0 and (token[0].isalpha() or token[0] in "_$") and all(c.isalnum() or c in "_$" for c in token)


def subtokenize(token: str) -> List[str]:
    """
    Subtokenizes an identifier into parts using CamelCase and snake_case rules.
    """
    if is_identifier(token):
        return RE_SUBTOKENIZE.split(token)
    else:
        return [token]


def subtokenize_batch(tokens: List[str]) -> Tuple[List[str], List[int]]:
    """
    Subtokenizes a list of tokens.
    Returns a tuple containing the list of subtokens and their corresponding original token indices.
    """
    sub_tokens = []
    src_indices = []
    for i, token in enumerate(tokens):
        new_sub_tokens = subtokenize(token)
        sub_tokens += new_sub_tokens
        src_indices += [i] * len(new_sub_tokens)
    return sub_tokens, src_indices


def get_tf_idf_query_similarity(corpus, query):
    """
    Computes cosine similarity between the query and each document in the corpus using TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    docs_tfidf = vectorizer.fit_transform(corpus)
    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    return cosineSimilarities


def run_TFIDF_baseline(project: str):
    """
    Processes evaluation data for the TFIDF baseline.
    Loads evaluation data, computes TF-IDF similarities, and writes results to a JSON file.
    """
    from seutil import IOUtils
    from pts.main import Macros  # Ensure Macros is updated in your pts/main.py
    eval_data_file = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag-qualifiedname.json"

    res = []
    evaldata = IOUtils.load(eval_data_file)
    for evalitem in evaldata:
        res_item = {}
        failed_test_list = evalitem["failed_test_list_qualified"]
        passed_test_list = evalitem["passed_test_list_qualified"]
        total_test_list = failed_test_list + passed_test_list
        labels = [1] * len(failed_test_list) + [0] * len(passed_test_list)
        changed_files = [
            file.lower().replace(".java", "")
            .replace("src/main/java/", "")
            .replace("src/test/java/", "")
            .replace(r"/", " ")
            for file in evalitem["changed_files"] if file.endswith(".java")
        ]
        # Tokenize each test name by splitting on "."
        corpus = [" ".join(subtokenize_batch(test.split("."))[0]).lower() for test in total_test_list]
        # Combine all changed files into one query string
        query = " ".join(changed_files)
        similarity = get_tf_idf_query_similarity(corpus, query).tolist()

        ekstazi_labels = [1 if test in evalitem["ekstazi_test_list"] else 0 for test in total_test_list]
        starts_labels = [1 if test in evalitem["starts_test_list"] else 0 for test in total_test_list]

        res_item["labels"] = labels
        res_item["prediction_scores"] = similarity
        res_item["commit"] = evalitem["commit"]
        res_item["Ekstazi_labels"] = ekstazi_labels
        res_item["STARTS_labels"] = starts_labels
        res.append(res_item)

    output_dir = Macros.model_data_dir / "rank-model" / project.split('_')[1] / "TFIDFBaseline" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    IOUtils.dump(output_dir / "per-sha-result.json", res, IOUtils.Format.jsonPretty)
