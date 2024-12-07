

GEMINI_KEYS = ['']




hyperparameter = { # iterations 1,2，3,4,5，10
 'hitab':{ # 1,2,4,6,8,10,20
    'LLM_select_cells':6,
    'dense_topk':2,
    'dense_score':0.9,
},
'ait-qa':{
    'LLM_select_cells':6,
    'dense_topk':2,
    'dense_score':0.9,
},
}

iterative_reasoning_system_instruction_english = """Suppose you are an expert in statistical analysis.
You will be given a table described in a special format. 
Your task is to answer the Question based on the content of the table. 

Graph Definition: We consider each cell in the table as a node in the graph, represented by the tuple (Row Index, Column Index, Cell Content). For example, (1, 0, "test1") represents the node in the 1st row and 0th column, with content "test1". The tuples (1, 0, "test1"), (1, 4, "test2"), and (3, 0, "test3") represent three nodes, where "test2" and "test1" have a SAME ROW relationship, "test3" and "test1" have a SAME COLUMN relationship, and "test1" is the shared neighbor of "test2" and "test3".
"""

iterative_reasoning_english = """You can interact with the graph through three steps: "Thought", "Action" and "Observation", to complete the question answering task step by step:
1. In the "Thought" step, thoroughly examine the question and the existing data. Determine if the current data is sufficient to answer the question:
    a. If the existing information is sufficient, proceed the "Action" step and call the "Answer" function to give the answer.
    b. If more information is needed, call the functions in the 'Action' step to obtain useful information.
2. In the "Action" step, you can call the following functions to get more node information from the graph :
    a. SearchNode(query): Retrieve the node from the graph that is semantically closest to the keyword 'query'(given as a str). Note: 'query' cannot be the known Cell Content, to avoid meaningless calls;
    b. GetAllNeighbours(node): Get all neighboring nodes in the same row and column of the specified node (given as a tuple) from the graph.
    c. GetSharedNeighbours(node1, node2): Get all shared neighbors between two specified nodes (also represented as tuples) in the graph. 
    d. Answer(): Answer the question based on the available information. 
"""


LLM_final_answer_english = """You MUST answer each question step by step as follows(Note: Keep your answer concise): 
1. Cell: The cells/nodes most relevant to the answer. 
2. Operation: the operation you performed on the cells/nodes you selected. 
3. Explanation: your explanation.
4. Answer: your final answer.
And if you need to extract relevant Cell Content from the graph as answer, do not add any units, symbols, or other explanatory text. Ensure that the extracted Content matches the original Cell Content in the graph exactly.""" # And if the answer is not contained within the context, say "I don’t know".

LLM_final_answer_format_english = """Please integrate all the current information to output your answer.
Using this JSON schema: Answer = {"cells" : list[str], "operation": str, "explanation": str, "answer": list[str] }. Return a `Answer`."""

THINK_FIRST_PROMPT_ENGLISH = """Please output your Thought step based on all the current information.
Note that if you need more information to answer the question, you must obtain more information by focusing on a particular cell in the current subgraph, i.e. you can only call the GetAllNeighbours function or GetSharedNeighbours function, and the function parameter must come from a cell in the current subgraph.
The output format is: "Thought step {step}: {your thought}" """

cot_think_english = """Let’s think step by step as follows and give full play to your expertise as a statistical analyst:
1. **Understand the Question**: Clearly understand the Question, clarify the relationships between the existing data, and organize the information needed to answer the Question.
2. **Analyze the Data Structure**: Have a comprehensive understanding of the data in the table/graph, including the meaning, data types, and formats of each cells/nodes. **Note:** Pay special attention to some **summative or aggregated cells/nodes** (e.g., "all", "combine", "total", "sum", "average", "mean",  "percent", "percentage", "proportion", "percent of total", "%", "probability", "likelihood", "% change", etc.), as these cells/nodes help you skip a lot of operations.
3. **Select Relevant Data**: Based on the Question, identify the most relevant cells/nodes. (**Note:** Some cells may have identical Cell Content, so avoid greedy searches when necessary; Focus on the most relevant and directly related data to the Question at hand.)
4. **Avoid Redundant Calculations**: Before performing any calculations or operations, first check if the needed information is already available in the table/graph. If so, directly use this information.
"""

cot_answer_english = """Let’s think step by step as follows and give full play to your expertise as a statistical analyst:
1. **Understand the Question**: Clearly understand the Question, clarify the relationships between the existing data, and organize the information needed to answer the Question. 
2. **Analyze the Data Structure**: Have a comprehensive understanding of the data in the table/graph, including the meaning, data types, and formats of each cells/nodes. **Note:** Pay special attention to some **summative or aggregated cells/nodes** (e.g., "all", "combine", "total", "sum", "average", "mean",  "percent", "percentage", "proportion", "percent of total", "%", "probability", "likelihood", "% change", etc.), as these cells/nodes help you skip a lot of operations.  
3. **Select Relevant Data**: Based on the Question, identify the most relevant cells/nodes.
4. **Avoid Redundant Calculations**: Before performing any calculations or operations, first check if the needed information is already available in the table/graph. If so, directly use this information.
5. **Synthesize the Answer**: Use the selected data to construct a clear and concise answer. Ensure that the final answer directly addresses the question, using the most relevant and accurate data from the table/graph.
"""

THINK_PROMPT_ENGLISH = """Please integrate all the current information to output your Thought step {step}. The output format is: "Thought step {step}: {your thought}" """


ACTION_PROMPT_ENGLISH = """Based on the results of your previous Thought step {step}, output your Action Step and Explanation.
Using this JSON schema: ActionStep = {"Function" : {"function_name": str, "parameters": list[tuple] | list[str] }, "Explanation": str}. Return a `list[ActionStep]`."""


start_cells_english = """The nodes which may help you answer the Question(help you locate evidence nodes related to the answer): [{start_cells}]."""

REASONING_PATH_PTOMPT_ENGLISH  = """{start_cells}

The steps below represent your interaction history on the graph. Please refer to them when selecting your next function. 
{reasoning_steps}"""

CONNECT_GRAPH_PROMPT_ENGLISH = """The subgraph below displays the neighbor relationships of the nodes/steps from your interaction history. This subgraph will dynamically update during your interaction process.
**Subgraph:**
{sub_graph}"""


GRAPH_DEFINITION_ENGLISH = """Graph Definition: We consider each cell in the table as a node in the graph, represented by the tuple (Row Index, Column Index, Cell Content). The graph contains two types of node relationships:
1. SAME ROW: undirected edge, representing that two cells are in the same row. Cells in the same column typically belong to the same attribute or category.
2. SAME COLUMN: undirected edge, representing that two cells are in the same column. Cells in the same row typically describe different attributes of the same object or instance.

For example, (1, 0, "test1") represents the cell in the 1st row and 0th column, with content "test1".
The tuples (1, 0, "test1"), (1, 4, "test2"), and (3, 0, "test3") represent three cells, where "test2" and "test1" have a SAME ROW relationship, "test3" and "test1" have a SAME COLUMN relationship, and "test1" is the shared neighbor of "test2" and "test3"."""
"""Graph Definition: We consider each cell in the table as a node in the graph, represented by the tuple (Row Index, Column Index, Cell Content). 
For example, (1, 0, "test1") represents the node in the 1st row and 0th column, with content "test1". The tuples (1, 0, "test1"), (1, 4, "test2"), and (3, 0, "test3") represent three nodes, where "test2" and "test1" have a SAME ROW relationship, "test3" and "test1" have a SAME COLUMN relationship, and "test1" is the shared neighbor of "test2" and "test3"."""


LLM_select_topk_cell_from_table_english = """Let’s think step by step as follows and give full play to your expertise as a statistical analyst: 
1. **Understand the Question**: Clearly understand the Question and the information needed to answer the Question to determine the necessary information to extract. 
2. **Analyze the Data Structure**: Have a comprehensive understanding of the data in the Table, including the meaning, data types, and formats of each cell tuples.    
3. **Select Relevant Data**: Based on the Question, identify the most relevant cell tuples. **Note:** Pay special attention to the header cell tuples in the Table, as they are often more relevant to the Question's semantics and can help in identifying the related evidence cell tuples.

{examples}

{table}

**Question:** {question}

Output format instructions:
1. Outputs cell tuples in descending order of relevance. 
2. Using this JSON schema: Tuple = {"tuple": tuple, "explanation": str}.  Return a `list[Tuple]`. 
"""


LLM_select_topk_cell_examples = """Here are some examples:
Table Caption: number of internet-enabled devices used per household member by household income quartile, households with home internet access, 2018 
Table:
(0, 0, '')	(0, 1, 'lowest quartile')	(0, 2, 'second quartile')	(0, 3, 'third quartile')	(0, 4, 'highest quartile')	(0, 5, 'total')
(1, 0, '')	(1, 1, 'percent')	(1, 2, 'percent')	(1, 3, 'percent')	(1, 4, 'percent')	(1, 5, 'percent')
(2, 0, 'less than one device per household member')	(2, 1, '63.0')	(2, 2, '60.7')	(2, 3, '56.9')	(2, 4, '56.2')	(2, 5, '58.4')
(3, 0, 'at least one device per household member')	(3, 1, '37.0')	(3, 2, '39.3')	(3, 3, '43.1')	(3, 4, '43.9')	(3, 5, '41.6')

Question: who were less likely to have less than one device per household member,households in the third quartile or those in the lowest quartile?
Select Cell Tuples: [{"tuple": (2, 0, "less than one device per household member"), "explanation": "This row shows the probability of households having less than one device per member, which we need to compare for the third and lowest quartiles."}, {"tuple": (0, 3, "third quartile"), "explanation": "This column represents the third quartile income group, for which we need the probability of having less than one device per member."}, {"tuple": (1, 3, "percent"), "explanation": "This cell confirms that the data in the third quartile column is in percentages, allowing for direct comparison of probabilities."}, {"tuple": (0, 1, "lowest quartile"), "explanation": "This column represents the lowest quartile income group, for which we need the probability of having less than one device per member."}, {"tuple": (1, 1, "percent"), "explanation": "This cell confirms that the data in the lowest quartile column is in percentages, allowing for direct comparison of probabilities."}]

Question: among households in the highest income quartile,what was the percentage of those who had less than one device per household member?
Select Cell Tuples: [{"tuple": (2, 0, "less than one device per household member"), "explanation": "This cell corresponds to the subset of households with less than one device per member, which is what the question asks about."}, {"tuple": (0, 4, "highest quartile"), "explanation": "This cell refers to the highest income quartile, which is the target group in the question."}, {"tuple": (1, 4, "percent"), "explanation": "This cell indicates that the data presented is in percentage format, aligning with how the question is phrased."}]

Question: among households in the lowest income quartile,what was the percentage of those who had less than one internet-enabled device for each household member?
Select Cell Tuples: [{"tuple": (0, 1, "lowest quartile"), "explanation": "This cell identifies the column representing the lowest income quartile, which is the target population of the question."}, {"tuple": (2, 0, "less than one device per household member"), "explanation": "This cell identifies the row corresponding to households with less than one device per member, which is what the question asks about."}, {"tuple": (1, 1, "percent"), "explanation": "This cell indicates that the values in the following rows of this column are percentages, the unit asked for in the question."}]

Question: among households who had internet access at home,what was the percentage of those who had less than one internet-enabled device per household member.
Select Cell Tuples: [{"tuple": (2, 0, "less than one device per household member"), "explanation": "This tuple is the row header for households with less than one internet-enabled device per household member."}, {"tuple": (0, 5, "total"), "explanation": "This tuple is the column header for the total percentage across all income quartiles."}, {"tuple": (1, 5, "percent"), "explanation": "This tuple is the row header for the percentage."}]

Table Caption: mental health indicators, by sexual orientation and gender, canada, 2018 
Table:
(0, 0, 'indicator')	(0, 1, 'heterosexual')	(0, 2, 'heterosexual')	(0, 3, 'heterosexual')	(0, 4, 'gay or lesbian')	(0, 5, 'gay or lesbian')	(0, 6, 'gay or lesbian')	(0, 7, 'bisexual')	(0, 8, 'bisexual')	(0, 9, 'bisexual')	(0, 10, 'sexual orientation n.e.c')	(0, 11, 'sexual orientation n.e.c')	(0, 12, 'sexual orientation n.e.c')	(0, 13, 'total sexual minority')	(0, 14, 'total sexual minority')	(0, 15, 'total sexual minority')
(1, 0, 'indicator')	(1, 1, 'percent')	(1, 2, '95% confidence interval')	(1, 3, '95% confidence interval')	(1, 4, 'percent')	(1, 5, '95% confidence interval')	(1, 6, '95% confidence interval')	(1, 7, 'percent')	(1, 8, '95% confidence interval')	(1, 9, '95% confidence interval')	(1, 10, 'percent')	(1, 11, '95% confidence interval')	(1, 12, '95% confidence interval')	(1, 13, 'percent')	(1, 14, '95% confidence interval')	(1, 15, '95% confidence interval')
(2, 0, 'indicator')	(2, 1, 'percent')	(2, 2, 'from')	(2, 3, 'to')	(2, 4, 'percent')	(2, 5, 'from')	(2, 6, 'to')	(2, 7, 'percent')	(2, 8, 'from')	(2, 9, 'to')	(2, 10, 'percent')	(2, 11, 'from')	(2, 12, 'to')	(2, 13, 'percent')	(2, 14, 'from')	(2, 15, 'to')
(3, 0, 'self-rated mental health')	(3, 1, '')	(3, 2, '')	(3, 3, '')	(3, 4, '')	(3, 5, '')	(3, 6, '')	(3, 7, '')	(3, 8, '')	(3, 9, '')	(3, 10, '')	(3, 11, '')	(3, 12, '')	(3, 13, '')	(3, 14, '')	(3, 15, '')
(4, 0, 'positive')	(4, 1, '88.9')	(4, 2, '88.3')	(4, 3, '89.4')	(4, 4, '80.3')	(4, 5, '75.5')	(4, 6, '84.4')	(4, 7, '58.9')	(4, 8, '52.4')	(4, 9, '65.2')	(4, 10, '54.6')	(4, 11, '37.4')	(4, 12, '70.9')	(4, 13, '67.8')	(4, 14, '63.6')	(4, 15, '71.7')
(5, 0, 'negative')	(5, 1, '10.7')	(5, 2, '10.2')	(5, 3, '11.2')	(5, 4, '19.7')	(5, 5, '15.6')	(5, 6, '24.4')	(5, 7, '40.9')	(5, 8, '34.7')	(5, 9, '47.4')	(5, 10, '45.4')	(5, 11, '29.1')	(5, 12, '62.6')	(5, 13, '32.1')	(5, 14, '28.2')	(5, 15, '36.3')
(6, 0, 'ever seriously contemplated suicide')	(6, 1, '14.9')	(6, 2, '14.3')	(6, 3, '15.5')	(6, 4, '29.9')	(6, 5, '25.1')	(6, 6, '35.1')	(6, 7, '46.3')	(6, 8, '39.8')	(6, 9, '52.8')	(6, 10, '58.7')	(6, 11, '42.1')	(6, 12, '73.5')	(6, 13, '40.1')	(6, 14, '36.1')	(6, 15, '44.3')
(7, 0, 'diagnosed mood or anxiety disorder')	(7, 1, '16.4')	(7, 2, '15.8')	(7, 3, '17.0')	(7, 4, '29.6')	(7, 5, '24.7')	(7, 6, '35.0')	(7, 7, '50.8')	(7, 8, '44.4')	(7, 9, '57.2')	(7, 10, '40.9')	(7, 11, '25.8')	(7, 12, '58.0')	(7, 13, '41.1')	(7, 14, '36.9')	(7, 15, '45.3')
(8, 0, 'mood disorder')	(8, 1, '9.5')	(8, 2, '9.1')	(8, 3, '10.0')	(8, 4, '20.6')	(8, 5, '16.4')	(8, 6, '25.4')	(8, 7, '36.2')	(8, 8, '30.0')	(8, 9, '42.9')	(8, 10, '31.1')	(8, 11, '18.0')	(8, 12, '48.1')	(8, 13, '29.1')	(8, 14, '25.3')	(8, 15, '33.3')
(9, 0, 'anxiety disorder')	(9, 1, '12.5')	(9, 2, '12.0')	(9, 3, '13.1')	(9, 4, '23.4')	(9, 5, '18.8')	(9, 6, '28.8')	(9, 7, '41.6')	(9, 8, '35.5')	(9, 9, '47.9')	(9, 10, '30.4')	(9, 11, '18.5')	(9, 12, '45.8')	(9, 13, '33.1')	(9, 14, '29.2')	(9, 15, '37.2')

Question: how many percent of heterosexual canadians have reported a mood or anxiety disorder diagnosis?
Select Cell Tuples: [{"tuple": (7, 0, "diagnosed mood or anxiety disorder"), "explanation": "This cell identifies the key indicator related to the question, which is the 'diagnosed mood or anxiety disorder'."},{"tuple": (1, 1, "percent"), "explanation": "This cell indicates that the values in Column 1 are presented as percentages, which is the measurement type the question asks for."},{"tuple": (0, 1, "heterosexual"), "explanation": "This cell identifies the demographic group 'heterosexual', which is the focus of the question."}]

Question: how many percent of sexual minority canadians have reported that they had been diagnosed with a mood or anxiety disorder?
Select Cell Tuples: [{"tuple": (7, 0, "diagnosed mood or anxiety disorder"), "explanation": "This cell contains the indicator 'diagnosed mood or anxiety disorder', which is central to the question."},{"tuple": (0, 13, "total sexual minority"), "explanation": "This cell identifies the 'total sexual minority' group, which is the population of interest in the question."},{"tuple": (1, 13, "percent"), "explanation": "This cell indicates that the data in this column is measured in 'percent', which is the specific measure the question asks for."}]

Question: how many percent of bisexual canadians and gay or lesbian canadians, respectively, have reported poor or fair mental health?
Select Cell Tuples: [{"tuple": (0, 7, "bisexual"), "explanation": "This cell indicates the column header related to 'bisexual' sexual orientation, relevant to the question."},{"tuple": (0, 4, "gay or lesbian"), "explanation": "This cell indicates the column header related to 'gay or lesbian' sexual orientation, relevant to the question."},{"tuple": (5, 0, "negative"), "explanation": "This cell indicates a 'negative' mental health status, which corresponds to poor or fair mental health, directly related to the question."},{"tuple": (1, 4, "percent"), "explanation": "This cell indicates that the column contains percentage values, which are necessary to answer the question about the percentage of gay or lesbian Canadians."},{"tuple": (1, 7, "percent"), "explanation": "This cell indicates that the column contains percentage values, which are necessary to answer the question about the percentage of bisexual Canadians."}]
"""



LLM_select_topk_cell_system_instruction_english = """Suppose you are an expert in statistical analysis.
You will be given a Table described in a special format.
Your task is to identify the cells in the Table that is most relevant to the Question.

Each cell in the Table is represented by a tuple (Row Index, Column Index, Cell Content). 
For example, the tuple (7, 0, "416") represents a cell at row 7, column 0, with a value of "416".
Make sure you read and understand these instructions carefully.
"""







shared_neighbors_english = """{cell_pair} has the following shared neighbors: {shared_cells}."""
no_shared_neighbors_but_same_row_english = """{cell_pair} has no shared neighbors, but the two nodes are on the SAME ROW."""
no_shared_neighbors_but_same_col_english = """{cell_pair} has no shared neighbors, but the two nodes are on the SAME COLUMN."""
no_shared_neighbors_english = """{cell_pair} has no shared neighbors."""

PROMPT_TEMPLATE = {
    'hitab':{
        'system_instruction': iterative_reasoning_system_instruction_english,
        'iterative_reasoning':iterative_reasoning_english,
        'graph_definition': GRAPH_DEFINITION_ENGLISH,
        'think_prompt':THINK_PROMPT_ENGLISH,
        'cot_think':cot_think_english,
        'cot_answer': cot_answer_english,
        'action_prompt':ACTION_PROMPT_ENGLISH,
        'connect_graph' : CONNECT_GRAPH_PROMPT_ENGLISH,
        'reasoning_path' : REASONING_PATH_PTOMPT_ENGLISH,
        'start_cells':start_cells_english,
        'LLM_select_cells':LLM_select_topk_cell_from_table_english,
        'LLM_select_cell_examples':LLM_select_topk_cell_examples,
        'LLM_select_cells_system_instruction': LLM_select_topk_cell_system_instruction_english,
        'LLM_final_answer': LLM_final_answer_english,
        'LLM_final_answer_format': LLM_final_answer_format_english,

        'shared_neighbors':shared_neighbors_english,
        'no_shared_neighbors_but_same_row': no_shared_neighbors_but_same_row_english,
        'no_shared_neighbors_but_same_col': no_shared_neighbors_but_same_col_english,
        'no_shared_neighbors': no_shared_neighbors_english,
    },
    'ait-qa': {
        'system_instruction': iterative_reasoning_system_instruction_english,
        'iterative_reasoning':iterative_reasoning_english,
        'graph_definition': GRAPH_DEFINITION_ENGLISH,
        'think_prompt':THINK_PROMPT_ENGLISH,
        'cot_think':cot_think_english,
        'cot_answer': cot_answer_english,
        'action_prompt':ACTION_PROMPT_ENGLISH,
        'connect_graph' : CONNECT_GRAPH_PROMPT_ENGLISH,
        'reasoning_path' : REASONING_PATH_PTOMPT_ENGLISH,
        'start_cells':start_cells_english,
        'LLM_select_cells':LLM_select_topk_cell_from_table_english,
        'LLM_select_cell_examples':LLM_select_topk_cell_examples,
        'LLM_select_cells_system_instruction': LLM_select_topk_cell_system_instruction_english,
        'LLM_final_answer': LLM_final_answer_english,
        'LLM_final_answer_format': LLM_final_answer_format_english,

        'shared_neighbors':shared_neighbors_english,
        'no_shared_neighbors_but_same_row': no_shared_neighbors_but_same_row_english,
        'no_shared_neighbors_but_same_col': no_shared_neighbors_but_same_col_english,
        'no_shared_neighbors': no_shared_neighbors_english,
    },

}



