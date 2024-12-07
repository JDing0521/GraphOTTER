
import ast
import re
from configs import PROMPT_TEMPLATE,GEMINI_KEYS,hyperparameter
from Generator.parse_output import get_action_list,parse_action_json, remove_quotes,LLM_json_output_parse
from collections import OrderedDict
import copy
from tools import table2markdown


class GraphReasoner:
    def __init__(self,args,model,query,table,caption,graph_retriever,dataset='hitab'):
        self.cot_think = ''
        self.system_instruction = ''

        self.dataset = dataset
        self.query = query
        self.table_caption = caption
        self.markdown_table = table
        self.user_input = []

        self.step = 1
        self.max_iteration_depth = args.max_iteration_depth

        self.output=''


        self.start_cells_prompt = ''
        self.args = args


        self.search_cell_history_dict = OrderedDict() # search 的结果
        self.Intermediate_results = OrderedDict() # 中间结果
        self.model = model
        self.graph_retriever = graph_retriever

        self.start_cells= ''
        self.connect_graph_prompt = '' # 子图prompt
        self.conn_graph_result = [] # 子图
        self.reasoning_path = []  #
        self.reasoning_path_dict = OrderedDict()  # 每一步的推理路径
        self.reasoning_path_prompt = '' # 推理路径prompt

    def initialize_prompt(self):
        cell_id_content_topk, cell_tuple_topk, conn_id_cell_dict, self.conn_graph_result = self.graph_retriever.initialize_subgraph(
            query=self.query)

        LLM_select_cells = list(cell_id_content_topk['LLM_select'].values())
        retriever_select_cells = list(cell_id_content_topk['retriever_select'].values())
        all_start_cells = LLM_select_cells + retriever_select_cells
        all_start_tuples = cell_tuple_topk['LLM_select'] + cell_tuple_topk['retriever_select']

        self.start_cells = PROMPT_TEMPLATE[self.dataset]['start_cells'].replace('{start_cells}', ','.join(
            ['{}'.format(i) for i in all_start_tuples]))

        self.connect_graph_prompt = PROMPT_TEMPLATE[self.dataset]['connect_graph'].replace('{sub_graph}', self.conn_graph_result)
        search_content = self.start_cells + '\n\n' + self.connect_graph_prompt
        print('###' * 20)
        print('检索到的单元格及其连通图', search_content)
        print('###' * 20)

        self.cot_think = PROMPT_TEMPLATE[self.dataset]['cot_think']

        self.system_instruction = PROMPT_TEMPLATE[self.dataset]['system_instruction']

        self.prompt = PROMPT_TEMPLATE[self.dataset]['iterative_reasoning'].replace('{question}', self.query)

        markdown_table = table2markdown(self.graph_retriever.dealed_rows)
        # markdown_table = table2Tuple(graph_retriever.dealed_rows)
        if self.table_caption:
            self.table_caption = 'Table Caption: {}\n'.format(self.table_caption)

        self.query = '**Question:** {}'.format(self.query)
        markdown_table = "**Table:**\n{}".format(markdown_table)
        self.markdown_table = self.table_caption + markdown_table

        self.retrieved_cell_id = list(cell_id_content_topk['LLM_select'].keys()) + list(
            cell_id_content_topk['retriever_select'].keys())
        self.retrieved_cell = copy.deepcopy(all_start_cells)
        self.hit_cell_neighbors_history_dict = OrderedDict()
        self.hit_cell_neighbors_history_dict[self.retrieved_cell[0]] = conn_id_cell_dict

    def LLM_generate(self,  prompt, isrepeated=0.0, response_mime_type=None):
        if 'gemini' not in self.model.model_name:
            result = self.model.generate(prompt, system_instruction=self.system_instruction, isrepeated=isrepeated,
                                    response_mime_type=response_mime_type)
        else:
            result = self.model.generate('\n'.join(prompt), system_instruction=self.system_instruction, isrepeated=isrepeated,
                                    response_mime_type=response_mime_type)
        return result
    
    def check_repeated_think_action(self, current_action_list, explanation_list,
                                    repeated_action, is_last_action_repeated, Thinking_text, Action_text):
        action_list = []
        result_action_list = []
        # 转换格式存储
        for j in range(len(current_action_list)):
            action = current_action_list[j]
            action_type, argument = parse_action_json(action)
            argument = '|'.join([str(i) for i in argument])
            if [action_type, argument] not in action_list:
                action_list.append([action_type, argument])
                result_action_list.append(action)
        for i in range(len(action_list)):
            action = action_list[i]
            for b_step, act in reversed(self.reasoning_path_dict.items()):
                if action in act:
                    if abs(self.step - b_step) > 1:
                        # repeated_action.append([step,temp_action_list[i],b_step])
                        repeated_action.append(
                            [self.step, f"{current_action_list[i]['function_name']}{str(current_action_list[i]['parameters'])}",
                             b_step])
                    else:
                        is_last_action_repeated = True
        if is_last_action_repeated or len(current_action_list) == 0:
            if is_last_action_repeated:
                prompt = copy.deepcopy(self.user_input[:-2]) + [
                    PROMPT_TEMPLATE[self.dataset]['think_prompt'].replace('{step}', str(self.step))]

                Thinking_text = self.LLM_generate(prompt, isrepeated=0.7)

                # Thinking_text = Thinking['text']
                Action_prompt = PROMPT_TEMPLATE[self.dataset]['action_prompt'].replace('{step}', str(self.step))

                prompt = prompt[:-1] + ['\n' + Thinking_text.strip().strip('\n'), '\n' + Action_prompt]
            else:
                print(str(self.step) + str(current_action_list) + 'Action 为空')
                prompt = copy.deepcopy(self.user_input)

            Action_text = self.LLM_generate(prompt, isrepeated=0.7)

            # Action_text = Action['text']


            result_action_list, explanation_list = get_action_list(Action_text)
            is_last_action_repeated = False

            return result_action_list, explanation_list, repeated_action, is_last_action_repeated , Thinking_text, Action_text

        return result_action_list, explanation_list, repeated_action, is_last_action_repeated, Thinking_text, Action_text

    def Thought(self):
        Thinking_prompt = PROMPT_TEMPLATE[self.dataset]['think_prompt'].replace('{step}', str(self.step))
        self.user_input = [
            self.prompt,
            self.cot_think,
            self.markdown_table,
            '\n' + self.query + '\n',
            self.start_cells + '\n' if self.step == 1 else self.reasoning_path_prompt + '\n',
            self.connect_graph_prompt,
        ]
        if self.step > 1:
            self.user_input += ['\n' + self.Intermediate_results[self.step - 1]['think'],
                                self.Intermediate_results[self.step - 1]['action'],
                                self.Intermediate_results[self.step - 1]['interaction_prompt']]
        self.user_input += ['\n' + Thinking_prompt]
        # self.user_input.append('\n'+Thinking_prompt)
        # iterative_process.append('\n'.join(self.user_input))

        if self.args.debug:
            print('Thought prompt：','\n'.join(self.user_input))

        # dialogs = [{"role": "user", "content": '\n'.join(self.user_input)}]

        Thinking = self.LLM_generate(self.user_input)
        return Thinking

    def Action(self,Thinking_text):
        Action_prompt = PROMPT_TEMPLATE[self.dataset]['action_prompt'].replace('{step}', str(self.step))

        self.user_input = [
            self.prompt,
            self.markdown_table,
            '\n' + self.query + '\n',
            self.start_cells + '\n' if self.step == 1 else self.reasoning_path_prompt + '\n',
            self.connect_graph_prompt,
        ]
        if self.step > 1:
            self.user_input += ['\n' + self.Intermediate_results[self.step - 1]['think'],
                                self.Intermediate_results[self.step - 1]['action'],
                                self.Intermediate_results[self.step - 1]['interaction_prompt']]
        self.user_input += ['\n' + Thinking_text.strip().strip('\n') + '\n', Action_prompt]

        print('---' * 30)
        if self.args.debug:
            print('\n'.join(self.user_input))
            print('---' * 30)

        Action = self.LLM_generate(self.user_input,
                                   response_mime_type="application/json" if 'gemini' in self.model.model_name else {
                                       "type": "json_schema"})
        return Action

    def get_cell_id(self,argument):
        if argument in self.retrieved_cell:  # 如果该cell是已经检索过的单元格
            cell_index = self.retrieved_cell.index(argument)
            cell_id = self.retrieved_cell_id[cell_index]
        else:
            is_arg_in_search_his = False
            if len(self.search_cell_history_dict) > 0:  # 如果该cell是所有检索过的单元格中的某个别称
                for hit_cell_id, his in self.search_cell_history_dict.items():
                    if is_arg_in_search_his:
                        is_arg_in_search_his = False
                        break
                    for k, v in his.items():
                        if k == argument:
                            is_arg_in_search_his = True
                            cell_id = hit_cell_id
                            break
            if len(self.hit_cell_neighbors_history_dict) > 0 and not is_arg_in_search_his:  # 如果该cell是所有检索过的单元格中的邻居单元格
                is_hit = False
                for _, neighbors in self.hit_cell_neighbors_history_dict.items():
                    if is_hit:
                        is_hit = False
                        break
                    for k, v in neighbors.items():
                        if v == argument:
                            cell_id = k
                            is_hit = True
                            break
        try:
            cell_id
        except NameError:
            cell_id_exist = False
            cell_id = None
        else:
            cell_id_exist = True
        return cell_id_exist,cell_id


    def Answer(self, Thinking_text, answer_explan='', last_interact_step=''):
        final_answer_prompt = PROMPT_TEMPLATE[self.dataset]['LLM_final_answer']
        cot_answer = PROMPT_TEMPLATE[self.dataset]['cot_answer']

        self.user_input = [
            cot_answer,
            final_answer_prompt + '\n',
            self.markdown_table,
            '\n' + self.query + '\n',
            self.start_cells + '\n',
            self.connect_graph_prompt,
        ]
        if self.step > 1:
            self.user_input += ['\n' + self.Intermediate_results[self.step - 1]['think'],
                           self.Intermediate_results[self.step - 1]['action'],
                           self.Intermediate_results[self.step - 1]['interaction_prompt']]
        self.user_input += ['\n' + Thinking_text.strip().strip('\n') + '\n' + answer_explan, last_interact_step,
                       PROMPT_TEMPLATE[self.dataset]['LLM_final_answer_format']]
        if self.args.debug:
            print('\n'.join(self.user_input))
        output = self.LLM_generate(self.user_input)

        print('LLM的回答', output)


        # answer = get_answer(output['text'])
        output = LLM_json_output_parse(output)
        answer = output['answer']
        # answer = answer_calculator(model, calculator, query, answer, dataset)

        return answer

    def filter_actions(self,current_action_list, explanation_list):

        if len(current_action_list) > 1:
            temp = []
            temp_ = []
            for i in range(len(current_action_list)):

                if 'answer' not in current_action_list[i]['function_name'].lower():
                    temp.append(current_action_list[i])
                    temp_.append(explanation_list[i])
            return temp,temp_
        return current_action_list, explanation_list

    def tuple2cell(self,argument):
        if (type(argument) == str and argument.startswith('(') and argument.endswith(')')) or type(
                argument) == tuple:
            nei_cell_tuple = ast.literal_eval(argument) if type(argument) == str else argument
            nei_cell_id_exist = True

            col_num = int(nei_cell_tuple[1])
            row_num = int(nei_cell_tuple[0])

            nei_cell_id = len(self.graph_retriever.dealed_rows[0]) * row_num + col_num
        else:
            nei_cell_id_exist, nei_cell_id = self.get_cell_id(argument)

        return nei_cell_id_exist, nei_cell_id

    def iterative_reasoning(self):

        self.initialize_prompt()

        repeated_action = []
        is_last_action_repeated = False
        while True:

            Thinking_text = self.Thought()

            # Thinking_text = Thinking['text']

            print('模型思考步骤为 {}'.format(Thinking_text))

            Action_text = self.Action(Thinking_text)
            # Action_text = Action['text'].replace(f'Action Step {self.step}:', '').replace(f'Action step {self.step}:', '')

            print('模型行动步骤为 {}'.format(Action_text))


            self.reasoning_path_dict[self.step] = []
            interaction_result = []

            # get action
            current_action_list, explanation_list = get_action_list(Action_text)

            current_action_list, explanation_list, repeated_action, \
            is_last_action_repeated, Thinking_text, Action_text = self.check_repeated_think_action(current_action_list,
                                                                                                    explanation_list,
                                                                                                    repeated_action,
                                                                                                    is_last_action_repeated,
                                                                                                    Thinking_text,
                                                                                                    Action_text)

            current_action_list, explanation_list = self.filter_actions(current_action_list, explanation_list)

            current_explanation_list = []
            if current_action_list:
                one_step_path = []
                for t_action in range(len(current_action_list)):
                    tmp_action = current_action_list[t_action]
                    try:
                        action_type, argument = parse_action_json(tmp_action)
                        tmp_action = f"{tmp_action['function_name']}({', '.join([str(i) for i in tmp_action['parameters']])})"
                    except Exception as e:
                        print(f'There is something wrong with the generated target actions {tmp_action}.')
                        raise Exception(f"Action解析错误 {e.__str__()}")
                    if action_type == 'Answer' or 'Answer' in action_type:
                        answer_explan_pattern = r'Explanation:(.*)'
                        match = re.search(answer_explan_pattern, explanation_list[t_action])
                        answer_explan = match.group(1).strip('\n') if match else explanation_list[t_action].strip('\n')
                        answer = self.Answer(
                            Thinking_text=Thinking_text, answer_explan=answer_explan)
                        # return answer, step, prompt_length, iterative_process

                        answer = ', '.join([str(i) for i in answer])
                        return answer

                    elif action_type == 'SearchNode':
                        # one_step_path.append(argument)
                        argument = str(argument[0])
                        search_cell, search_cell_id, search_cell_tuple = self.graph_retriever.search_cell(query=argument,
                                                                                                     topk=1)
                        if len(search_cell_id) > 0:
                            self.search_cell_history_dict[search_cell_id[0]] = {argument: search_cell[0]}

                        # interaction_result.append([tmp_action, search_cell])
                        interaction_result.append([tmp_action, search_cell_tuple])
                        self.reasoning_path_dict[self.step].append([action_type, '|'.join([str(i) for i in [argument]])])

                        one_step_path.append(tmp_action)
                        current_explanation_list.append(explanation_list[t_action])

                        if len(search_cell) > 0 and search_cell[0] not in self.retrieved_cell:
                            self.retrieved_cell.append(search_cell[0])
                            self.retrieved_cell_id.append(search_cell_id[0])
                            hit_cell_same_row_col = self.graph_retriever.getSameRow_ColCells(match_cell_id=self.retrieved_cell_id)
                            _, _, self.conn_graph_result = self.graph_retriever.hit_cell_connect_graph(hit_cell_same_row_col)

                    elif action_type == 'GetAllNeighbours':
                        argument = argument[0]

                        nei_cell_id_exist, nei_cell_id = self.tuple2cell(argument)

                        if not nei_cell_id_exist:
                            nei_cell_id, nei_cell_id_exist = self.graph_retriever.check_arg_exists(argument)
                        if not nei_cell_id_exist:
                            print('GetAllNeighbours 没找到cell', argument)
                            raise Exception(f'GetAllNeighbours 没找到cell {argument}')
                        else:
                            cell_topk, nei_cells, hit_cell_neighbors_content_id = self.graph_retriever.get_neighbors(
                                add_id_list=[nei_cell_id],
                                get_same_row=False,
                                get_all_nei=True)
                            self.hit_cell_neighbors_history_dict[nei_cell_id] = hit_cell_neighbors_content_id

                            interaction_result.append([tmp_action, '\n' + nei_cells])
                            self.reasoning_path_dict[self.step].append([action_type, '|'.join([str(i) for i in [argument]])])
                            one_step_path.append(tmp_action)
                            current_explanation_list.append(explanation_list[t_action])

                            if len(cell_topk) > 0 and cell_topk[0] not in self.retrieved_cell:
                                self.retrieved_cell.append(cell_topk[0])
                                self.retrieved_cell_id.append(nei_cell_id)
                                hit_cell_same_row_col = self.graph_retriever.getSameRow_ColCells(
                                    match_cell_id=self.retrieved_cell_id)
                                _, _, self.conn_graph_result = self.graph_retriever.hit_cell_connect_graph(hit_cell_same_row_col)
                            del nei_cell_id
                    elif action_type == 'GetSharedNeighbours':
                        try:
                            cell1, cell2 = argument if len(argument) == 2 else argument[:2]
                            cell1 = remove_quotes(cell1) if type(cell1) == str else cell1
                            cell2 = remove_quotes(cell2) if type(cell2) == str else cell2
                        except Exception as e:
                            print('GetSharedNeighbours 解析报错', argument)
                            raise Exception(f'GetSharedNeighbours 解析报错 {argument}')
                        else:
                            cell1_exists, cell1_id = self.tuple2cell(cell1)
                            cell2_exists, cell2_id = self.tuple2cell(cell2)

                            if not cell1_exists:
                                cell1_id, cell1_exists = self.graph_retriever.check_arg_exists(cell1)
                            if not cell2_exists:
                                cell2_id, cell2_exists = self.graph_retriever.check_arg_exists(cell2)
                            if not cell1_exists or not cell2_exists:

                                print('GetSharedNeighbours 没找到cell', argument)
                                raise Exception(f'GetSharedNeighbours 没找到cell {argument}')
                            else:
                                hit_cell_same_row_col = self.graph_retriever.getSameRow_ColCells(
                                    match_cell_id=[cell1_id, cell2_id])
                                _, connect_id_cell_dict, cell_shared_neighbors = self.graph_retriever.hit_cell_connect_graph(
                                    hit_cell_same_row_col, get_shared_nei=True)

                                interaction_result.append([tmp_action, '\n' + cell_shared_neighbors])
                                self.reasoning_path_dict[self.step].append([action_type, '|'.join([str(i) for i in argument])])
                                one_step_path.append(tmp_action)
                                current_explanation_list.append(explanation_list[t_action])

                                self.hit_cell_neighbors_history_dict[cell1_id] = connect_id_cell_dict
                self.reasoning_path.append(str(one_step_path))

            interaction_result = '\n'.join(
                ['{}. The result of {} is: {}'.format(i + 1, interaction_result[i][0], interaction_result[i][1]) for i
                 in
                 range(len(interaction_result))])

            print('****' * 20)
            print('查询结果为：', interaction_result)
            print('****' * 20)

            if len(repeated_action) > 0 and not is_last_action_repeated:
                repeated_action_prompt = PROMPT_TEMPLATE[self.dataset]['repeated_action_attention'].replace('{step}',
                                                                                                       str(self.step)) \
                    .replace('{action}', ','.join([str(i[1]) for i in repeated_action])) \
                    .replace('{last_step}', ','.join(list(set([str(i[2]) for i in repeated_action]))))
                Interaction_prompt = """Observation Step {}:\n{}\n\n{}""".format(str(self.step), interaction_result,
                                                                                 repeated_action_prompt)
                repeated_action = []
            else:
                Interaction_prompt = """Observation Step {}:\n{}""".format(str(self.step), interaction_result)

            self.Intermediate_results[self.step] = {
                'think': Thinking_text.strip().strip('\n').replace('\n', ''),
                'action': f"Action Step {self.step}:\n" + '\n'.join(
                    ['{}. {}'.format(i + 1, current_explanation_list[i]) for i in range(len(current_explanation_list))]),
                'interaction_prompt': Interaction_prompt,

            }

            reasoning_steps = '\n'.join(
                ['Step {}:'.format(i + 1) + str(self.reasoning_path[i]) for i in range(len(self.reasoning_path))])
            self.reasoning_path_prompt = PROMPT_TEMPLATE[self.dataset]['reasoning_path'].replace('{start_cells}', self.start_cells) \
                .replace('{reasoning_steps}', reasoning_steps)

            self.connect_graph_prompt = PROMPT_TEMPLATE[self.dataset]['connect_graph'].replace('{sub_graph}', self.conn_graph_result)

            if self.step > self.max_iteration_depth:
                print("迭代次数超过{}次，该次推理必须回答".format(self.max_iteration_depth))

                answer = self.Answer(Thinking_text=Thinking_text,last_interact_step=f"{self.Intermediate_results[self.step]['action']}\n{self.Intermediate_results[self.step]['interaction_prompt']}\n")
                answer = ', '.join([str(i) for i in answer])

                return answer
            self.step += 1
