from urllib.parse import urljoin
import requests
import time 
from argparse import Namespace
# from web_search.bing_search import bing_web_search
# from web_search.bing_search import extract_relevant_info

import asyncio
from openai import OpenAI
# from deep_search_dgt import deep_search_dgt
from transformers import AutoTokenizer






def debug_code_function(code, error):

    API_BASE_URL = "xxxxxx"  
    MODEL_NAME = "Qwen2.5-72B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )

    prompt = f"""You are a code expert. I need you to debug the following code. Below are the code originally generated by the model and the error information that occurred during code execution. Please output ONLY the corrected Python code, without any explanation or markdown formatting:

    **Inputs:**

    **Original Code:**
    {code}

    **Execution Error:**
    {error}

    Output the corrected Python code only, without any explanation or markdown formatting:
    """

    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    
    response_text = chat_response.choices[0].message.content
    
    # 提取Python代码
    if "```python" in response_text and "```" in response_text:
        # 如果代码被markdown格式包围，提取其中的代码
        code_start = response_text.find("```python") + 9
        code_end = response_text.find("```", code_start)
        extracted_code = response_text[code_start:code_end].strip()
    else:
        # 如果没有markdown格式，直接使用返回的内容
        extracted_code = response_text.strip()
    # import pdb
    # pdb.set_trace()
    print("提取的代码:", extracted_code)
    return extracted_code






if __name__ == "__main__":

    code = """
# 创建符号变量
x, y = symbols('x y')

# 定义方程
eq1 = x**2 + y**2 = 25  # 错误的等号使用
eq2 = 2*x + y = 10

# 求解方程组
solution = solve((eq1, eq2), (x, y))
print(f"Solution: {solution}")
    """
    
    error = "SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '=' ?"
    
    # 调用debug函数
    debug_code_function(code, error)
    