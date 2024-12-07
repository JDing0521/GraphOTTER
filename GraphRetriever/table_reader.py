import json
import os
import copy
import jsonlines
import xlrd
import pandas as pd
import csv

def im_tqa_excelReader(file):
    # table_list = []
    # for file in os.listdir(excel_folder):
    file_name = os.path.splitext(file)[0]
    # 打开Excel
    wb = xlrd.open_workbook(file)
    sheets = wb.sheet_names()
    assert len(sheets) == 1

    sheet = wb.sheet_by_name(sheets[0])
    # 获取总行数
    rows_num = sheet.nrows
    # 获取总列数
    cols_num = sheet.ncols

    merged_cells = sheet.merged_cells

    dealed_rows = []
    rows = []
    for ri in range(rows_num):
        row_cells = sheet.row_values(ri)
        rows.append(sheet.row_values(ri))
        for rlo, rhi, clo, chi in merged_cells:
            # 判断行下标是否在合并单元格范围内
            if ri in range(rlo, rhi) and chi > clo + 1:
                for coli in range(clo, chi):
                    row_cells[coli] = sheet.cell_value(rlo, clo)
            # 判断行下标是否在合并单元格范围内
            if ri in range(rlo, rhi) and rhi > rlo + 1:
                for coli in range(clo, chi):
                    row_cells[coli] = sheet.cell_value(rlo, clo)
        # row_cells = ','.join([repr(i) for i in row_cells if i])
        dealed_rows.append(row_cells)
    dealed_cols = []
    cols = []
    for ci in range(cols_num):
        col_cells = sheet.col_values(ci)
        cols.append(sheet.col_values(ci))
        for rlo, rhi, clo, chi in merged_cells:
            # 判断行下标是否在合并单元格范围内
            if ci in range(clo, chi) and rhi > rlo + 1:
                for rowi in range(rlo, rhi):
                    col_cells[rowi] = sheet.cell_value(rlo, clo)

            if ci in range(clo, chi) and chi > clo + 1:
                for rowi in range(rlo, rhi):
                    col_cells[rowi] = sheet.cell_value(rlo, clo)
        # col_cells = ','.join([repr(i) for i in col_cells if i])
        dealed_cols.append(col_cells)

        # table_list = [{file_name :{ 'rows':rows, 'cols':cols}}]

    return dealed_rows,dealed_cols,rows,cols,merged_cells

def hitab_table_converter(table_path):
    with open(table_path) as f:
        table = json.load(f)
    merged_cells = []
    table['texts'] = [[c.lower() for c in r ] for r in table['texts'] ]
    table_text = copy.deepcopy(table['texts'])
    for merged_cell in table['merged_regions']:
        rlo,rhi,clo,chi = merged_cell['first_row'],merged_cell['last_row'],merged_cell['first_column'],merged_cell['last_column']
        merged_r, merged_c = rhi + 1, chi+1
        if rhi >= len(table_text):
            merged_r = len(table_text)
        if chi >= len(table_text[0]):
            merged_c = len(table_text[0])
        merged_cells.append((rlo, merged_r, clo, merged_c))

        for r in range(rlo,rhi+1):
            for c in range(clo,chi+1):
                try:
                    table_text[r][c] = table_text[rlo][clo]
                except IndexError:
                    print(table_path,len(table_text),len(table_text[0]),r,c,'数据集原始索引有问题')
                    continue
    cols = []
    dealed_cols = []
    for j in range(len(table['texts'][0])):
        temp = []
        dealed_temp = []
        for i in range(len(table['texts'])):
            temp.append(table['texts'][i][j])
        cols.append(temp)
        for i in range(len(table_text)):
            dealed_temp.append(table_text[i][j])
        dealed_cols.append(dealed_temp)
    return table_text,dealed_cols,table['texts'],cols,merged_cells

def crt_qa_table_converter(table_path):
    table = pd.read_csv(table_path,delimiter='#')
    table_list = []
    row_index = table.keys().to_list()
    table_list.append(row_index)
    for row in table.iterrows():
        index,col = row
        col = col.to_list()

        table_list.append(col)
    cols = []
    for j in range(len(table_list[0])):
        temp = []
        for i in range(len(table_list)):
            temp.append(table_list[i][j])
        cols.append(temp)

    return table_list,cols,table_list,cols,[]

def wtq_table_converter(table_path):
    table_list = []
    with open(table_path) as f:
        max_row_length = 0
        for row in csv.reader(f):
            table_list.append(row)
            if len(row) > max_row_length:
                max_row_length = len(row)
    table_list = [row+['']*(max_row_length-len(row)) for row in table_list]

    cols = []
    for j in range(len(table_list[0])):
        temp = []
        for i in range(len(table_list)):
            temp.append(table_list[i][j])
        cols.append(temp)

    return table_list,cols,table_list,cols,[]

def im_tqa_load_table(table_path):
    table_id_dict = {}
    with open(table_path, "r+",encoding='utf-8') as f:
        tables = json.load(f)
    for table in tables:
        cell_ID_matrix = table['cell_ID_matrix']
        english_cell_value_list = []
        for i in cell_ID_matrix:
            temp = []
            for j in i:
                # temp.append(table['chinese_cell_value_list'][j])
                temp.append(table['english_cell_value_list'][j])

            english_cell_value_list.append(temp)
        table['organized_english_cell_value_list'] = english_cell_value_list
        table_id_dict[table['table_id']] = table
    return table_id_dict
def im_tqa_tableId2Answer(id_list, table):
    content_dict = {}
    # row_id = id // len(table[0])
    # col_id = id % len(table[0])
    # content = table[row_id][col_id]
    for r in range(len(table['cell_ID_matrix'])):
        for c in range(len(table['cell_ID_matrix'][r])):
            for id in id_list:
                if table['cell_ID_matrix'][r][c] == id :
                    # content_dict[id] = table['organized_chinese_cell_value_list'][r][c]
                    content_dict[id] = table['organized_english_cell_value_list'][r][c]
    return list(content_dict.values())

def im_tqa_tableId2Content(table):

    table_list = table['organized_english_cell_value_list']
    cols = []
    for j in range(len(table_list[0])):
        temp = []
        for i in range(len(table_list)):
            temp.append(table_list[i][j])
        cols.append(temp)
    table_path = '/public/mm24/datasets/IM-TQA/data/Excel_tables/' + table['file_name'] + '.xlsx'
    # table_path = '/data/qianlong_data/Code/Document_understanding/TU/Datsets/IM-TQA/data/Excel_tables/' + table['file_name'] + '.xlsx'
    wb = xlrd.open_workbook(table_path)
    sheets = wb.sheet_names()
    sheet = wb.sheet_by_name(sheets[0])
    merged_cells = sheet.merged_cells

    return table_list, cols, table_list, cols, merged_cells

def ait_qa_converter(qa):
    table = qa['table']
    cols = []
    for j in range(len(table[0])):
        temp = []
        for i in range(len(table)):
            temp.append(table[i][j])
        cols.append(temp)
    return table, cols, table, cols, []

def TableJsonReader(json_path,dataset='aitqa'):

    if dataset == 'hitab':
        table_list = []
        for file in os.listdir(json_path):
            file_name = os.path.splitext(file)[0]
            with open(json_path + file,'r') as f:
                table_json = json.load(f)
            table_texts = table_json['texts']
            if len(table_json['merged_regions']) > 0:
                for merge_cell in table_json['merged_regions']:
                    first_row = merge_cell['first_row']
                    last_row = merge_cell['last_row']
                    first_column = merge_cell['first_column']
                    last_column = merge_cell['last_column']
                    for row_index in range(first_row,last_row+1):
                        for col_index in range(first_column,last_column+1):
                            table_texts[row_index][col_index] = table_texts[first_row][first_column]

            col_max_depth = 0
            for item in table_texts:
                if len(item) > col_max_depth:
                    col_max_depth = len(item)
            cols = []
            for j in range(col_max_depth):
                col = []
                for i in range(len(table_texts)):
                    col.append(table_texts[i][j])
                cols.append(col)

            table_list.append({file_name :{ 'rows':table_texts, 'cols':cols}})


    elif dataset == 'aitqa':
        with open(json_path,'r',encoding='utf-8') as f:
            tables_list = [json.loads(line) for line in f ]
        table_list = []
        for table in tables_list:

            table_matrix = []
            col_max_depth = 0
            for item in table['column_header']:
                if len(item) > col_max_depth:
                    col_max_depth = len(item)
            row_max_depth = 0
            for item in table['row_header']:
                if len(item) > row_max_depth:
                    row_max_depth = len(item)

            for k in range(col_max_depth):
                if len(table['row_header']) > 0:
                    temp = ['']*row_max_depth
                else:
                    temp = []
                for c_h in table['column_header']:
                    if len(c_h) > k:
                        temp.append(c_h[k])
                table_matrix.append(temp)

            for i in range(len(table['data'])):
                if len(table['row_header']) > 0:
                    temp = table['row_header'][i] + table['data'][i]
                else:
                    temp = table['data'][i]
                table_matrix.append(temp)

            cols = []
            for k in range(row_max_depth):
                col = ['']*col_max_depth
                for i in range(len(table['row_header'])):
                    r = table['row_header'][i]
                    if len(r) > k:
                        col.append(r[k])
                    else:
                        col.append('')
                cols.append(col)
            for j in range(len(table['column_header'])):
                col = []
                col.append(''.join(table['column_header'][j]))
                for i in range(len(table['data'])):
                    r = table['data'][i]
                    col.append(r[j])
                cols.append(col)

            table_list.append({table['id'] :{ 'rows':table_matrix, 'cols':cols}})
    return table_list




if __name__ == '__main__':
    # json_path = '/public/mm24/datasets/CRT-QA/CRT-QA/all_csv/1-1921-1.html.csv'
    # table_list,cols,table_list,cols,_ = crt_qa_table_converter(json_path)
    # print(table_list)
    # print(cols)
    qa_path = '/public/mm24/datasets/WikiTableQuestions/pristine-unseen-tables.tsv'
    wtq_qas = pd.read_csv(qa_path, sep='\t')

    for temp_qa in wtq_qas.iterrows():
        index, qa = temp_qa
        qa = qa.to_list()
        query = qa[1]
        id = qa[0]
        table_path =  qa[2]
        answer = qa[3]
        print(id,table_path,answer)

