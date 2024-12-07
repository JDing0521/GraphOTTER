
import re
import string
import time



def maybe_normalize_float(span: str):
    if span and (re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
                 or (re.match(r"^[0-9]*[.]?[0-9]*$", span))) and span != '.':
        # FIXME: We did this(instead of try except) to convert a string into a float
        #  since the try catch will lead to an error when using 8 V100 gpus with cuda 11.0,
        #  and we still don't know why that could happen....
        return str(float(span))
    else:
        return span


def maybe_normalize_number(text: str) -> str:
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    for index, unit in enumerate(units):
        if text == unit:
            return str(float(index))
    return text

def normalize_false_true(text: str) -> str:
    if text.lower() == "false":
        return "no"
    elif text.lower() == "true":
        return "yes"
    else:
        return text

import unicodedata

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()

def remove_punc(text: str) -> str:
    # Step 1: Remove inner content of parentheses using regular expressions
    text = re.sub(r'\([^)]*\)', '', text).strip()
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)




def remove_articles(text: str) -> str:
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def remove_quotes(s):
    s = s.strip().strip('\n').strip().strip('\n').strip('(').strip(')').strip('（').strip('）')
    if s.startswith(("'", '"','‘','’','“','”')):
        s = s[1:]
    if s.endswith(("'", '"','‘','’','“','”')):
        s = s[:-1]
    return s



def eval_ex_match(pred, gold_result):
    pred = remove_quotes(pred.lower())
    gold_result = str(gold_result).lower()
    compare_1 = pred.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_1) == 0:
        compare_1 = " "
    compare_2 = gold_result.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_2) == 0:
        compare_2 = " "
    if compare_1[0] == "-":
        compare_1 = compare_1[1:]
    if compare_2[0] == "-":
        compare_2 = compare_2[1:]
    if (
            compare_1.isdigit() == True
            and compare_2.isdigit() == True
            and pred.count(".") < 2
            and gold_result != "-"
    ):
        if pred[-1] == ".":
            pred = pred[0: len(pred) - 1]
        gold_result = gold_result.replace(",", "").replace("%", "")
        if gold_result[-1] == '.':
            gold_result = gold_result[:-1]
        pred = pred.replace(",", "").replace("%", "")
        pred = abs(float(pred))
        gold_result = abs(float(gold_result))
        if abs(pred - gold_result) < 0.01:
            return True
        else:
            return False

    # Replace and with comma
    if 'and' in pred and '|' in gold_result:
        pred = pred.replace('and', ',')

    pred = [span.strip() for span in pred.split(',') if span.strip()]

    if '|' in gold_result:
        # gold_result = [span.strip() for span in gold_result.split('|')]
        gold_result = gold_result.replace('|', ',')
    # else:
    gold_result = [span.strip() for span in gold_result.split(',') if span.strip()]

    pred = [normalize_unicode(normalize_false_true(maybe_normalize_number(remove_punc(remove_articles(span.strip().strip('.0')))))) for span in pred]
    gold_result = [normalize_unicode(normalize_false_true(maybe_normalize_number(remove_punc(remove_articles(span.strip().strip('.0')))))) for span in gold_result]
    # print(pred, ' # ', gold_result)
    clean_float = True  # TODO
    if clean_float:
        pred = [maybe_normalize_float(span) for span in pred]
        gold_result = [maybe_normalize_float(span) for span in gold_result]

    res = 1 if sorted(pred) == sorted(gold_result) else 0


    return res #or gold_result[0] in pred[0]



prompt_template = """Instruction: You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no.

You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth Answer" and the "Prediction" to determine whether the prediction correctly answers the question. 
All information in the Ground-truth Answer must be present in the Prediction, including numbers and dates. 
You must answer "no" if there are any specific details in the Ground-truth Answer that are not mentioned in the Prediction. 
There should be no contradicting statements in the Prediction. The Prediction may contain extra information. If the Prediction states something as a possibility, treat it as a definitive answer.

Question: {question}
Ground-truth Answer: {ground_truth}
Prediction:  {response}

CompareGPT response:"""

def LLM_eval(model,query,response,answer):
    pred = remove_quotes(response.lower())
    gold_result = str(answer).lower()
    compare_1 = pred.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_1) == 0:
        compare_1 = " "
    compare_2 = gold_result.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_2) == 0:
        compare_2 = " "
    if compare_1[0] == "-":
        compare_1 = compare_1[1:]
    if compare_2[0] == "-":
        compare_2 = compare_2[1:]
    if (
            compare_1.isdigit() == True
            and compare_2.isdigit() == True
            and pred.count(".") < 2
            and gold_result != "-"
    ):
        if pred[-1] == ".":
            pred = pred[0: len(pred) - 1]
        gold_result = gold_result.replace(",", "").replace("%", "")
        if gold_result[-1] == '.':
            gold_result = gold_result[:-1]
        pred = pred.replace(",", "").replace("%", "")
        pred = abs(float(pred))
        gold_result = abs(float(gold_result))
        if abs(pred - gold_result) < 0.01:
            return True,'',''
        else:
            return False,'',''

    if '|' in answer:
        # gold_result = [span.strip() for span in gold_result.split('|')]
        answer = answer.replace('|', ', ')
    answer = answer.lower()
    response = response.lower()

    prompt = prompt_template.replace('{question}',query).replace('{ground_truth}',answer).replace('{response}',response)

    res = model.generate(prompt)
    time.sleep(1.0)

    res = res.lower().strip().strip('\n')
    # print(query,'###',answer,'###',response,'###',res)
    if 'yes' in res:
        return 1
    else:
        return 0



