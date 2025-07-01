import ast
import requests
import pandas as pd
from copy import deepcopy
import logging
import yaml

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = '8842bd84-06cb-494a-b59d-4be5cac36214'

MODE = 'text/event-stream'
HEADERS = {
    'Authorization': API_KEY,
    'Content-Type': 'application/json;charset=UTF-8'
}

BODY_TMPLT = {
    'model': 'Chatrhino-81B-Pro',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': '介绍一下你自己'}
    ],
    'temperature': 0.7
}

URL = 'http://gpt-proxy.jd.com/gateway/baidu/chat/completions'

class MyRequest:
    def __init__(self, model='Chatrhino-81B-Pro'):
        self.body = deepcopy(BODY_TMPLT)
        self.body['model'] = model

    def send(self, system_prompt, user_prompt):
        self.body['messages'][0]['content'] = system_prompt
        self.body['messages'][1]['content'] = user_prompt
        try:
            logging.info('Sending request to API...')
            resp = requests.post(URL, headers=HEADERS, json=self.body)
            resp.raise_for_status()  # 检查请求是否成功，抛出异常
            resp_msg = dict(resp.json()['choices'][0]['message'])
            logging.info('Request successful.')
            return resp_msg['content']
        except requests.exceptions.HTTPError as http_err:
            logging.error(f'HTTP error occurred: {http_err}')  # 记录HTTP错误
            return ''
        except Exception as e:
            logging.error(f'An error occurred: {e}')  # 记录其他错误
            return ''

def generate_variations_from_excel(
        input_excel_file='国补兜底话术hr.xlsx', 
        output_yaml_file='generalized_sentences.yaml',
        problem_type='problem_type', 
        sentences='sentences', 
        generalization_times=10):
    """
    从 Excel 文件读取句子，生成变体，并保存到 YAML 文件。

    参数：
    input_excel_file (str): 输入的 Excel 文件名。
    output_yaml_file (str): 输出的 YAML 文件名。
    problem_type (str): 第一列的列名，用作键。
    sentences (str): 包含句子的列名。
    generalization_times (int): 为每个句子生成变体的数量。
    """
    # 读取 Excel 文件
    try:
        df = pd.read_excel(input_excel_file)  # 根据需要调整文件名
    except Exception as e:
        logging.error(f'读取 Excel 文件时出错: {e}')
        return

    # 前向填充以处理第一列中的合并单元格
    df[problem_type] = df[problem_type].ffill()

    # 创建一个字典以存储合并结果
    merged_results = {}

    for index, row in df.iterrows():
        req = MyRequest('gpt-4.1')
        
        key = str(row.get(problem_type, '')).strip()  # 获取问题类型并去除空格
        original_sentence = str(row.get(sentences, '')).strip()  # 获取句子并去除空格
        
        logging.info(f'处理第 {index + 1} 行: {original_sentence}')
        
        # 检查填充后的键是否为空
        if not key or not original_sentence:
            logging.warning(f'第 {index + 1} 行的键或句子为空. 正在跳过...')
            continue
        
        try:
            generalized_sentence = req.send(
                f"你是一个智能ai客服，同时也是专业的语句泛化专家，你要尊重用户你的回答不能过于轻浮。下面的是你可能会回复用户的一些语句，"
                f"对于下面这些语句请帮我换种方式表述出来，但是要和原来的语句的意思保持一致，"
                f"请尝试从正式专业风、文艺温柔风、活泼热情风等多种风格进行改写，需要进行{generalization_times}次改写，回复的内容中不需要有序号和具体风格，"
                f"你的输出是必须是列表格式，即['结果1', '结果2', '结果3', ..., '结果{generalization_times}',...]",
                original_sentence
            )
            generalized_sentences = ast.literal_eval(generalized_sentence)
            
            # 初始化字典中的键
            if key not in merged_results:
                merged_results[key] = []
                
            # 仅保存变体到键中
            merged_results[key].extend(generalized_sentences)
            logging.info(f'第 {index + 1} 行生成的句子: {generalized_sentences}')
            
        except Exception as e:
            logging.error(f'处理第 {index + 1} 行时出错: {e}')

    # 保存结果为 YAML
    save_results_as_yaml(merged_results, output_yaml_file)

def save_results_as_yaml(merged_results, output_yaml_file):
    """
    将合并结果字典保存为 YAML 文件。

    参数：
    merged_results (dict): 包含合并结果的字典。
    output_yaml_file (str): 输出的 YAML 文件名。
    """
    class CustomDumper(yaml.Dumper):
        def represent_scalar(self, tag, value, **options):
            # 将字符串始终用双引号表示
            if isinstance(value, str):
                return super().represent_scalar(tag, value, style='"', **options)
            return super().represent_scalar(tag, value, **options)

    # 保存到 YAML 文件
    try:
        with open(output_yaml_file, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(merged_results, yaml_file, Dumper=CustomDumper, allow_unicode=True, sort_keys=False)
        logging.info(f'YAML 输出已保存到 {output_yaml_file}。')
    except Exception as e:
        logging.error(f'保存到 YAML 文件时出错: {e}')

if __name__ == "__main__":
    # 这里可以初始化您想要的文件名和参数
    input_excel_file = '国补兜底话术hr.xlsx'  # 输入的 Excel 文件名
    output_yaml_file = '国补兜底话术gen.yaml'  # 输出的 YAML 文件名
    problem_type = '类型'  # 问题类型列名
    sentences = 'sentences'         # 要泛化的句子列名ls
    generalization_times = 10        # 要生成的变体数量

    generate_variations_from_excel(
        input_excel_file=input_excel_file,
        output_yaml_file=output_yaml_file,
        problem_type=problem_type,
        sentences=sentences,
        generalization_times=generalization_times
    )
