import ast
import os
import re
import json

def split_checks(input_string):
    # pattern = r'[\w]+\(.*?\)'
    pattern = r'[\w]+\[.*?\]'
    # Use re.findall to get all matches
    result = re.findall(pattern, input_string)
    return result
def output_json_parser(input_string):
    input_string = input_string.strip().strip('\n').strip().replace('\n','')
    json_match = re.search(r'```json(.+)```', input_string, re.DOTALL)
    json_match2 = re.search(r'\[.+\]', input_string)

    json_match3 = re.search(r'\{.+\}', input_string)

    match_str = []
    if json_match:
        json_string = json_match.group(1)
        match_str = ast.literal_eval(json_string.strip().strip('\n'))
    elif input_string.strip('```').strip('"').strip("'").strip().startswith('[') and input_string.strip(
            '```').strip('"').strip("'").strip().endswith(']'):
        match_str = ast.literal_eval(input_string.strip('```').strip('"').strip("'").strip())
    elif json_match2:
        json_str = json_match2.group()
        # 将JSON字符串转换为列表
        match_str = ast.literal_eval(json_str)
    if len(match_str) ==0 or type(match_str[0]) != dict or "Function" not in match_str[0].keys():
        if input_string.strip('```').strip('"').strip("'").strip().startswith('{') and input_string.strip(
                '```').strip('"').strip("'").strip().endswith('}'):
            match_str = ast.literal_eval(input_string.strip('```').strip('"').strip("'").strip())
            match_str = [match_str]
        elif json_match3:
            json_str = json_match3.group()
            # 将JSON字符串转换为列表
            match_str = ast.literal_eval(json_str)
            match_str = [match_str]
        else:
            return [],[]


    func_list = []
    exlanation_lsit = []
    for match in match_str:
        try:
            func_list.append(match['Function'])
            exlanation_lsit.append(
                f"Function: {match['Function']['function_name']}({', '.join([str(i) for i in match['Function']['parameters']])}), Explanation: {match['Explanation']}")
        except KeyError as k:
            print('LLM action 输出解析报错',k.__str__())
            raise Exception(f'LLM action 输出解析报错 {k.__str__()}')
            # continue
    return func_list, exlanation_lsit

def get_action_list(string):
    # if string[:len('Finish')] == 'Finish':
    #     return [string]
    # else:
    #     # return string.split(', ')
    # return split_checks(string)
    return output_json_parser(string)



def parse_action_json(function_dict):
    action_type = function_dict['function_name']
    argument = function_dict['parameters']
    parameters = []
    for item in argument:
        if type(item) == list and len(item) > 0 and (type(item[0]) == tuple or (type(item[0]) == str and item[0].startswith('(') and item[0].endswith(')') and item[0].count(',') >= 2)):
            for it in item:
                parameters.append(ast.literal_eval(it) if type(it) == str and it.startswith('(') and it.endswith(')') and it.count(',') >= 2 else it )
        elif type(item) == list and 'answer' in (action_type.lower()):
            for it in item:
                parameters.append(it)
        elif type(item) == list:
            parameters.append(tuple(item))
        elif type(item) == str and  item.startswith('(') and item.endswith(')') and item.count(',') >= 2:
            try:
                parameters.append(ast.literal_eval(item))
            except Exception as e:
                print('LLM输出action无法解析',item,e.__str__())
                item_list = [i.strip().strip('(').strip(')').strip('"').strip("'").strip() for i in item.split(',')]
                parameters.append((int(item_list[0]),int(item_list[1]),''.join(item_list[2:])))
        else:
            parameters.append(item)
    return action_type,parameters

def LLM_json_output_parse(output):
    output = output.strip().strip('\n').strip().replace('\n','')
    json_match = re.search(r'```json(.+)```', output, re.DOTALL)
    # 使用正则表达式提取JSON
    json_pattern = r'\{.+\}'
    json_match2 = re.search(json_pattern, output)
    json_match3 = re.search(r'\[.+\]', output)
    if json_match:
        json_string = json_match.group(1)
        try:
            result = ast.literal_eval(json_string.strip().strip('\n'))
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif output.strip('```').strip('"').strip("'").strip().startswith('{') and output.strip(
            '```').strip('"').strip("'").strip().endswith('}'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif output.strip('```').strip('"').strip("'").strip().startswith('[') and output.strip(
            '```').strip('"').strip("'").strip().endswith(']'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif json_match2:
        json_str = json_match2.group()
        # 将JSON字符串转换为字典
        result = ast.literal_eval(json_str)
    elif json_match3:
        json_str = json_match3.group()
        # 将JSON字符串转换为字典
        result = ast.literal_eval(json_str)
    else:
        raise UserWarning('LLM输出解析报错 ' + output)

    return result
def remove_quotes(s):
    s = s.strip().strip('\n')
    if s.startswith(("'", '"','‘','’','“','”')):
        s = s[1:]
    if s.endswith(("'", '"','‘','’','“','”')):
        s = s[:-1]
    return s



