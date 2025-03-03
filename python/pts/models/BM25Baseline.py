from rank_bm25 import BM25Okapi
import string
from seutil import BashUtils, IOUtils
from typing import List
from pts.main import Macros
try:
    import javalang
except ModuleNotFoundError:
    print("Warning: javalang not available; falling back to simple tokenization.")
    javalang = None
from functools import reduce
from pts.models.TFIDFbaseline import get_tf_idf_query_similarity


def max_abs_normalize(scores):
    """Normalize the scores"""
    max_score = max(scores)
    return [s / max_score for s in scores]


def getBM25sims(trainDocs, query):
    """
    Given a list of training documents (as strings) and a query (as a string),
    build a BM25 model using rank_bm25 and return the similarity scores as a list.
    """
    # Tokenize each document: BM25Okapi expects a list of token lists.
    tokenized_corpus = [doc.split() for doc in trainDocs]
    bm25model = BM25Okapi(tokenized_corpus)
    # Tokenize the query
    sims = bm25model.get_scores(query.split())
    return sims.tolist()  # Convert numpy array to list for JSON serialization


def build_bm25_model(trainDocs):
    """
    Given a list of training documents (as strings), build and return a BM25 model.
    """
    tokenized_corpus = [doc.split() for doc in trainDocs]
    return BM25Okapi(tokenized_corpus)


def tokenize(s):
    """
    Custom tokenization function that processes a string and returns a space-separated string of tokens.
    """
    result = ""
    buffer = ""
    for word in s.split():
        if word.isupper():
            if len(word) > 1:
                result += word.lower() + " "
        else:
            for c in word:
                if c in string.ascii_lowercase:
                    buffer += c
                elif c in string.ascii_uppercase:
                    if buffer != "":
                        if len(buffer) > 1:
                            result += buffer + " "
                        buffer = ""
                    buffer += c.lower()
                else:
                    if buffer != "":
                        if len(buffer) > 1:
                            result += buffer + " "
                        buffer = ""
            if buffer != "":
                if len(buffer) > 1:
                    result += buffer + " "
                buffer = ""
    return result


def parse_file(SHA: str, filepath: str):
    counter = {}
    try:
        with open(filepath) as f:
            content = f.read().replace("\n", " ")
            if javalang is not None:
                try:
                    tokens = javalang.tokenizer.tokenize(content)
                    for token in tokens:
                        name = type(token).__name__
                        if name == 'Operator' or "Integer" in name or "Floating" in name or name == 'Separator':
                            continue
                        else:
                            counter[token.value] = counter.get(token.value, 0) + 1
                except Exception as e:
                    print(f"Error tokenizing with javalang in {filepath}: {e}")
                    # fallback to simple tokenization
                    for token in content.split():
                        counter[token] = counter.get(token, 0) + 1
            else:
                # Fallback if javalang isn't available
                for token in content.split():
                    counter[token] = counter.get(token, 0) + 1
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    # Combine tokens into a string (each token repeated according to its count)
    return reduce(lambda x, key: (key + " ") * counter[key] + x, counter, "")


def pre_proecessing_for_each_sha(project, eval_data_item, subset="All"):
    SHA = eval_data_item["commit"][:8]
    if subset == "All":
        failed_test_list = eval_data_item["failed_test_list"]
        passed_test_list = eval_data_item["passed_test_list"]
        total_test_list = failed_test_list + passed_test_list
    elif subset == "Ekstazi":
        total_test_list = eval_data_item["ekstazi_test_list"]
    elif subset == "STARTS":
        total_test_list = eval_data_item["starts_test_list"]

    changed_files = eval_data_item["diff_per_file"].keys()

    project_folder = Macros.repos_downloads_dir / f"{project}"
    if not project_folder.exists():
        BashUtils.run(f"git clone https://github.com/{project.replace('_', '/')} {project_folder}")

    test_content = {}
    change_content = {}
    with IOUtils.cd(project_folder):
        BashUtils.run(f"git checkout {SHA}")
        for test in total_test_list:
            filepath = BashUtils.run(f"find . -name {test}.java").stdout
            if filepath == "":
                continue
            if len(filepath.split("\n")) > 1:
                filepath = filepath.split("\n")[0]
            test_content[test] = parse_file(SHA, filepath)

        for changed_file in changed_files:
            change_content[changed_file] = parse_file(SHA, changed_file)
        eval_data_item["data_objects"] = test_content
        eval_data_item["queries"] = change_content
    return eval_data_item


def pre_processing(project: str):
    eval_data_json = IOUtils.load(Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag.json")
    res = []
    for eval_data_item in eval_data_json:
        eval_data_item = pre_proecessing_for_each_sha(project, eval_data_item)
        res.append(eval_data_item)
    res_path = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag-preprocessed.json"
    IOUtils.dump(res_path, res, IOUtils.Format.jsonPretty)


def run_BM25_baseline_for_each_sha(eval_data_item, rule=False, subset="All"):
    """Run BM25 baseline for the given SHA and return the results in the form of a dictionary."""
    res_item = {}
    changed_files = eval_data_item["diff_per_file"].keys()
    failed_test_list = eval_data_item["failed_test_list"]
    passed_test_list = eval_data_item["passed_test_list"]
    if subset == "All":
        test_list = failed_test_list + passed_test_list
    elif subset == "Ekstazi":
        test_list = eval_data_item["ekstazi_test_list"]
    elif subset == "STARTS":
        test_list = eval_data_item["starts_test_list"]
    else:
        raise NotImplementedError
    if not test_list:
        res_item["labels"] = []
        res_item["prediction_scores"] = []
        res_item["commit"] = eval_data_item["commit"]
        res_item["Ekstazi_labels"] = []
        res_item["STARTS_labels"] = []
        return res_item

    # Tokenize test file contents using the custom tokenize function.
    trainDocs = [tokenize(eval_data_item["data_objects"].get(i, "")) for i in test_list]
    # Combine and tokenize all query strings (changed files).
    query = tokenize(" ".join([eval_data_item["queries"].get(i, "") for i in changed_files]))
    BM25sims = getBM25sims(trainDocs, query)

    ekstazi_labels = []
    starts_labels = []
    labels = []
    for test in test_list:
        ekstazi_labels.append(1 if test in eval_data_item["ekstazi_test_list"] else 0)
        starts_labels.append(1 if test in eval_data_item["starts_test_list"] else 0)
        labels.append(1 if test in failed_test_list else 0)
    res_item["labels"] = labels
    if rule:
        BM25sims = max_abs_normalize(BM25sims)
        changed_classes = [t.split("/")[-1].replace(".java", "") for t in eval_data_item["diff_line_number_list_per_file"].keys()]
        modified_test_class = [t for t in changed_classes if "Test" in t]
        for index, t in enumerate(test_list):
            if t in modified_test_class:
                BM25sims[index] = 1
    res_item["prediction_scores"] = BM25sims
    res_item["commit"] = eval_data_item["commit"]
    res_item["Ekstazi_labels"] = ekstazi_labels
    res_item["STARTS_labels"] = starts_labels
    return res_item


def run_BM25_baseline(project: str):
    processed_eval_data_json = IOUtils.load(Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag-preprocessed.json")
    res = []
    for eval_data_item in processed_eval_data_json:
        res_item = run_BM25_baseline_for_each_sha(eval_data_item)
        res.append(res_item)
    output_dir = Macros.model_data_dir / "rank-model" / project.split('_')[1] / "BM25Baseline" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    IOUtils.dump(output_dir / "per-sha-result.json", res, IOUtils.Format.jsonPretty)


def pre_proecessing_for_training_sha(project, changed_file_list: List[str]):
    """
    Extract the contents for the PIT tool mutants, test files' contents, and source files' contents.
    Returns a dictionary mapping file paths to their contents.
    """
    from pts.main import proj_logs
    training_SHA = proj_logs[project]
    data_item = {}

    collected_results_dir = Macros.repos_results_dir / project / "collector"
    if (collected_results_dir / "method-data.json").exists():
        test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")
    else:
        raise FileNotFoundError("Cannot find test2meth.json.")
    total_test_list = list(test_class_2_methods.keys())

    project_folder = Macros.repos_downloads_dir / f"{project}"
    if not project_folder.exists():
        BashUtils.run(f"git clone https://github.com/{project.replace('_', '/')} {project_folder}")

    test_content = {}
    change_content = {}
    with IOUtils.cd(project_folder):
        BashUtils.run(f"git checkout {training_SHA}")
        for test in total_test_list:
            filepath = BashUtils.run(f"find . -name {test}.java").stdout
            if filepath == "":
                print(test, "filepath is empty")
            if len(filepath.split("\n")) > 1:
                filepath = filepath.split("\n")[0]
            test_content[test] = parse_file(training_SHA, filepath)

        for changed_file in changed_file_list:
            change_content[changed_file] = parse_file(training_SHA, changed_file)
        data_item["data_objects"] = test_content
        data_item["queries"] = change_content
    return data_item


def get_BM25_score(data_item, changed_file: str, bm25_model):
    query = tokenize(data_item["queries"].get(changed_file, ""))
    BM25sims = bm25_model.get_scores(query.split())
    return BM25sims
