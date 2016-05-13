import re


# Functions to load babi dataset

# Returns a list of (input, question, answer) tuples for the training set
def get_task_1_train():
    return get_babi_dataset('../data/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt')

# Returns a list of (input, question, answer) tuples for the training set
def get_task_1_test():
    return get_babi_dataset('../data/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt')

# Returns a list of (input, question, answer) tuples for the training set
def get_task_6_train():
    return get_babi_dataset('../data/babi/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt')

# Returns a list of (input, question, answer) tuples for the training set
def get_task_6_test():
    return get_babi_dataset('../data/babi/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_test.txt')

# Returns a list of (input, question, answer) tuples
def get_babi_dataset(path):
    print "Loading babi data"
    babi_data_path = path

    babi_data = open(babi_data_path).read()
    lines = data_to_list(babi_data)

    data, story = [], []
    for line in lines:
        nid, line = line.strip().split(' ', 1)

        # Check for start of a new story
        if int(nid) == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            substory = [x for x in story if x]
            data.append((substory, tokenize(q), a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)

    return [(flatten(_story), _question, answer) for _story, _question, answer in data]


def tokenize(sentence):
    """
    Split a sentence into tokens including punctuation.
    Args:
        sentence (string) : String of sentence to tokenize.
    Returns:
        list : List of tokens.
    """
    return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]


def flatten(data):
    """
    Flatten a list of data.
    Args:
        data (list) : List of list of words.
    Returns:
        list : A single flattened list of all words.
    """
    return reduce(lambda x, y: x + y, data)


def data_to_list(data):
    """
    Clean a block of data and split into lines.
    Args:
        data (string) : String of bAbI data.
    Returns:
        list : List of cleaned lines of bAbI data.
    """
    split_lines = data.split('\n')[:-1]
    return [line.decode('utf-8').strip() for line in split_lines]
