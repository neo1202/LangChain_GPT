# -*- coding: utf-8 -*-
"""1810.04805v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BKeir4RmgVm9Zp4c30KOqHEy5NBTEdkQ

## 套件、環境
"""

"""
import os
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/Colab/富邦') #切換該目錄
os.listdir() #確認目錄內容
"""
# Commented out IPython magic to ensure Python compatibility.
# always needed
import math, os, random, csv
from config import OPEN_API_KEY, PINECONE_KEY, SERP_API_KEY
#from torch.utils.tensorboard import SummaryWriter
from math import gamma
from tabnanny import verbose
import pandas as pd
import numpy as np
# log and save
import json, logging, pickle, sys, shutil, copy
from argparse import ArgumentParser, Namespace
from pathlib import Path
from copy import copy
import joblib

# %matplotlib inline
import seaborn as sns
# others
import matplotlib.pyplot as plt
from PIL import Image

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader

import jieba as jb

import torch
from langchain.vectorstores import Pinecone
import pinecone
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain import OpenAI
from langchain.agents import AgentType
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
import nltk
nltk.download('punkt')
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Union
import zipfile
import re
from langchain import SerpAPIWrapper, LLMChain, LLMMathChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory 

from langchain.document_loaders import WebBaseLoader
from serpapi import GoogleSearch

from langchain.chat_models import ChatOpenAI
from text2vec import SentenceModel
global_llm_chat, global_embeddings = None, None

def initialize():
    global global_llm_chat, global_embeddings
    os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
    os.environ["SERPAPI_API_KEY"] = SERP_API_KEY
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

    #OpenAI類默認對應 「text-davinci-003」版本：
    #OpenAIChat類默認是 "gpt-3.5-turbo"版本
    #OpenAI是即將被棄用的方法，最好是用ChatOpenAI
    model = SentenceModel('shibing624/text2vec-base-chinese')
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese', model_kwargs={'device': EMBEDDING_DEVICE})  #768維度
    llm_chat = ChatOpenAI(temperature=0) #GPT-3.5-turbo
    llm_chat
    global_llm_chat, global_embeddings = llm_chat, embeddings
    return llm_chat, embeddings

def process_and_store_documents(file_paths: List[str]) -> None:
    global global_embeddings
    def init_txt(file_pth: str):
        loader = TextLoader(file_pth)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separators=[" ", ",", "\n", "\n\n", "\t", ""]
        )
        split_docs_txt = text_splitter.split_documents(documents)
        return split_docs_txt
    
    def init_csv(file_pth: str):
        # my_csv_loader = CSVLoader(file_path=f'{file_pth}',encoding="utf-8", 
        #                           csv_args={'delimiter': ','
        # })
        loader = DirectoryLoader(f'{file_pth}', glob='**/*.csv', loader_cls=CSVLoader, silent_errors=True)
        documents = loader.load()
        split_docs_csv = documents #這份csv資料已經人為切割
        return split_docs_csv
    
    def init_xlsx(file_pth: str):
        loader = UnstructuredExcelLoader(file_pth,mode="elements")
        split_docs_xlsx = loader.load() 
        return split_docs_xlsx
    
    def init_pdf(file_pth: str):
        loader = PyPDFLoader(file_pth)
        split_docs_pdf = loader.load_and_split()
        return split_docs_pdf

    def init_word(file_pth: str):
        loader = Docx2txtLoader(file_pth)
        split_docs_word = loader.load()
        return split_docs_word
    
    def init_ustruc(file_pth: str):
        loader = UnstructuredFileLoader(file_pth)
        split_docs_ustruc = loader.load()
        return split_docs_ustruc
    
    pinecone.init(
    api_key=PINECONE_KEY,
    environment="us-west1-gcp-free"
    )
    index_name="demo-langchain" #768 #open ai embedding為1536向量

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine', #or dotproduct
            dimensions=768
        )
    index = pinecone.Index(index_name)
    index.describe_index_stats()
    #print(f"Processing file paths: {file_paths}")
    #if not index.describe_index_stats()['total_vector_count']:
    doc_chunks = []
    for file_path in file_paths:
        txt_docs = init_txt(file_path)
        #print(f"Document chunks from {file_path}: {txt_docs}")
        doc_chunks.extend(txt_docs)
    Pinecone.from_texts([t.page_content for t in doc_chunks], global_embeddings, index_name=index_name)
    docsearch=Pinecone.from_existing_index(index_name,global_embeddings) #傳入當初embedding的方法
    index_stats = index.describe_index_stats()
    #print(index_stats)
    
"""## 模板 （Agent, tool, chain)

## 定義Tools的集合
"""
def get_my_agent():
    global global_embeddings
    pinecone.init(
    api_key= PINECONE_KEY,
    environment="us-west1-gcp-free"
    )
    index_name="demo-langchain" 
    index = pinecone.Index(index_name)
    index.describe_index_stats() 
    docsearch = Pinecone.from_existing_index(index_name,global_embeddings)
    CONTEXT_QA_Template = """
    根據以下提供的信息，回答用戶的問題
    信息：{context}

    問題：{query}
    """
    CONTEXT_QA_PROMPT = PromptTemplate(
        input_variables=["context", "query"],
        template=CONTEXT_QA_Template,
    )
    class FugeDataSource:
        def __init__(self, llm:ChatOpenAI(temperature=0.2)):
            self.llm = llm

        def find_product_description(self, product_name: str) -> str:
            """模拟公司产品的数据库"""
            product_info = {
                "好快活": "好快活是一個營銷人才平台，以社群+公眾號+小程序結合的運營模式展開，幫助企業客戶連接並匹配充滿才華的營銷人才。",
                "Rimix": "Rimix通過採購流程數字化、完備的項目數據存儲記錄及標準的供應商管理體系，幫助企業實現採購流程, 透明合規可追溯，大幅節約採購成本。Rimix已為包括聯合利華，滴滴出行等多家廣告主提供服務，平均可為客戶節約採購成本30%。",
                "Bid Agent": "Bid Agent是一款專為中國市場設計的搜索引擎優化管理工具，支持5大搜索引擎。Bid Agent平均為廣告主提升18%的投放效果，同時平均提升47%的管理效率。目前已為陽獅廣告、GroupM等知名4A公司提供服務與支持。",
            }
            return product_info.get(product_name, "没有找到这个产品")

        def find_company_info(self, query: str) -> str:
            """模擬公司介紹文檔數據庫，讓llm根據抓取信息回答問題"""
            context = """
            關於產品："讓廣告技術美而溫暖"是復歌的產品理念。在努力為企業客戶創造價值的同時，也希望讓使用復歌產品的每個人都能感受到技術的溫度。
            我們關注用戶的體驗和建議，我們期待我們的產品能夠給每個使用者的工作和生活帶來正面的改變。
            我們崇尚技術，用科技的力量使工作變得簡單，使生活變得更加美好而優雅，是我們的願景。
            企業文化：復歌是一個非常年輕的團隊，公司大部分成員是90後。
            工作上，專業、注重細節、擁抱創新、快速試錯。
            協作中，開放、坦誠、包容、還有一點點舉重若輕的幽默感。
            以上這些都是復歌團隊的重要特質。
            在復歌，每個人可以平等地表達自己的觀點和意見，每個人的想法和意願都會被尊重。
            如果你有理想，並擁有被理想所驅使的自我驅動力，我們期待你的加入。
            """
            prompt = CONTEXT_QA_PROMPT.format( context=context, query=query )
            return self.llm(prompt = prompt)
    fuge_data_source = FugeDataSource(global_llm_chat) #初始化

    """#### 公司內部文檔搜尋QA
    包含score
    """

    #### 分為兩段：
    # - 第一段：得到前k筆資料是有價值的(score大於某門檻)
    # - 第二段：讓retrievalQA去搜尋前k筆資料並依據其作出回答

    prompt_template_fubon = """
    你是個專業文檔師，你的任務是從給定的上下文回答問題
        你的回答應該基於我提供的資訊回答我的問題，並以對話的方式呈現。
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        問題如下：
        {question}
        
        給定的資訊：
        {context}
        
        請綜合上述信息,你給出的回復需要包含以下3個字段:
        1.text: 用於存放你總結性的文字回復,儘量完整
        2.similiarAnswers: 基於我提的問題與上下文, 聯想2個我可能會想瞭解的不同維度的問題

        請按照以下JSON格式來回答:

        前括號
        "text": "<這裡放你的回答>",
        "similiarAnswers": [
            "<聯想的問題1>",
            "<聯想的問題2>"
        ]
        後括號

    """
    prompt_fubon = PromptTemplate(
        template=prompt_template_fubon, input_variables=["context", "question"]
        #context即是向量搜尋取回的文件們
    )
    class FubonDataSource:
        def __init__(self, llm:OpenAI(temperature=0)):
            self.llm = llm
        def find_doc_above_score(self, query: str) -> str:
            """讓chain知道前幾筆資料是有用的, pass到search.kwarg 因pinecone+langchain不支援同時取score"""
            model = SentenceModel('shibing624/text2vec-base-chinese')
            query_embedd = model.encode(query, convert_to_numpy=True).tolist()
            response = index.query(query_embedd, top_k=2, include_metadata=True)
            #print(response) 
            threshold = 0.60
            above_criterion_cnt = 0
            for data in response['matches']:
                if data['score'] < threshold:
                    break;
                print(data)
                above_criterion_cnt += 1
            print(f"\nHow many docs match the criterion? {above_criterion_cnt} docs\n")
            return above_criterion_cnt
        def return_doc_summary(self, query: str) -> str:
            k = self.find_doc_above_score(query)

            if k == 0: return '沒有內部相符的文檔'
            data_retriever = RetrievalQA.from_chain_type(llm=retrieval_llm, 
                                            chain_type="map_reduce", 
                                            retriever= docsearch.as_retriever(search_kwargs={"k": k}),
                                            chain_type_kwargs = {"verbose": False,
                                                                "question_prompt": prompt_fubon, #注意是question_prompt
                                                                },
                                            return_source_documents=False)
            return data_retriever.run(query)

    retrieval_llm = ChatOpenAI(temperature=0)
    fubon_data_source = FubonDataSource(retrieval_llm) #初始化

    """#### 搜尋數篇網路文章並總結"""

    def sumWebAPI(input_query: str) : 
        '''依照關鍵字搜尋google前n個網址並總結'''

        num_news = 2 # 找前2篇網站
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=10, separators=[" ", ",", "\n", "\n\n", "\t", ""]
        )

        # refine方法的template
        prompt_template = """Write a concise summary about 100 words of the following:

        {text}

        CONCISE SUMMARY IN Chinese:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        refine_template = (
            "Your job is to produce a final summary so that a reader will have a full understanding of what happened\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary in Chinese around 300 words"
            "If the context isn't useful, return the original summary."
        )
        refine_prompt = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )
        # Google search Api params
        params = {
        "q": f"{input_query}",
        "location": "Taiwan",
        "hl": "tw", #國家
        "gl": "us",
        "google_domain": "google.com",
        # your api key
        "api_key": "06089eea6970e557b98953b8a61cbbb3747c0b8651a8c331faba9dbbc166c9a3",
        "num": f"{num_news}"
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        ### Get Website, title ###
        title_news, link_news = [], []
        for i in range(len(results['organic_results'])):
            title_news.append(results['organic_results'][i]['title'])
            link_news.append(results['organic_results'][i]['link'])
        print(f"related top {num_news} title: ", title_news)
        print(f"websites: ", title_news)

        loader = WebBaseLoader(link_news) #can switch to other loader
        documents = loader.load() #網站都合在一起變成document
        #print(documents)
        def extract_text(document):
            content = document.page_content.strip()
            content = re.sub(r'[\n\t]+', '', content)
            text = re.sub(r'[^\w\s.]', '', content)
            return text
        
        split_docs = text_splitter.split_documents(documents)
        # 取部分內容作總結就行
        if len(split_docs) >= 10:
            split_docs = split_docs[3:9]
        elif len(split_docs) > 5:
            split_docs = split_docs[1:5]

        for doc in split_docs:
            doc = extract_text(doc)
            print('here is a doc:', doc)
        chain = load_summarize_chain(global_llm_chat, chain_type="refine", question_prompt=PROMPT, refine_prompt=refine_prompt
                                    ,verbose=False) #verbose可以看過程
        result = chain.run(split_docs)
        return result

    """#### 其餘小工具"""

    search = SerpAPIWrapper(params = {'engine': 'google', 'gl': 'us', 'google_domain': 'google.com', 'hl': 'tw'})
    llm_math_chain = LLMMathChain(llm=global_llm_chat, verbose=False)

    # from langchain.docstore.document import Document
    # def summarizeText(input_text: str) : 
    #     '''單純的總結一段文字'''
    #     #每一段原始text
    #     map_prompt_template = """Write a concise summary of the following to 30 to 50 words:


    #     {text}


    #     CONCISE SUMMARY IN Chinese:"""
    #     map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    #     #最後總結那些精簡後的text
    #     combine_prompt_template = """Write a concise summary of the following to 80 to 160 words:


    #     {text}


    #     CONCISE SUMMARY IN Chinese:"""
    #     combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    #     text_splitter = RecursiveCharacterTextSplitter(
    #       chunk_size=1000, chunk_overlap=20, separators=[" ", ",", "\n", "\n\n", "\t", ""]
    #     )
    #     texts = text_splitter.split_text(input_text)
    #     # Create Document objects for the texts
    #     docs = [Document(page_content=t) for t in texts]

    #     chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce",
    #                                  return_intermediate_steps=False, map_prompt=map_prompt, combine_prompt=combine_prompt)
    #     summary = chain.run(docs)
    #     return summary

    """### 將Tools 集合丟給Agent調用"""

    customize_tools = [
        #Tool(
         #   name = "SimpleSearchWeb",
          #  func=search.run,
           # description="Only use when you need to answer simple questions about current events after 2022"
        #),
        Tool(
            name = '查詢相關資訊',
            func=fubon_data_source.return_doc_summary,
            description="Useful for questions related to topics that they upload to get more information,\
            your action input here must be a single sentence query that correspond to the question"
        ),
        Tool(
            name = "SummarizeWebInformation",
            func=sumWebAPI,
            description="Only use when you need to conclude web information after 2022, input should be key word"
        ),
        # Tool(
        #     name = "SummarizeTextInput",
        #     func=summarizeText,
        #     description="Useful when you want to summarize a piece of text, regardless of its length, input should be a string of text"
        # ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name="查詢復歌科技公司產品名稱",
            func=fuge_data_source.find_product_description,
            description="通过产品名称找到复歌科技产品描述时用的工具，输入应该是产品名称",
        ),
        Tool(
            name="復歌科技公司相關信息",
            func=fuge_data_source.find_company_info,
            description="幫用戶詢問復歌科技公司相关的问题, 可以通过这个工具了解相关信息",
        )
    ]
    """## 初始化Agent with Tools"""
    class CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # 解析 llm 的輸出，根據輸出文本找到需要執行的決策。
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                #print(f'找到最後答案了，此次的llm_output為: \n{llm_output}')
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            
            # Parse out the action and action input
            regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL) #DOTALL代表可以是任何字元
            
            # If it can't parse the output it raises an error
            if not match:
                raise ValueError(f"暫時無法解析您的問題。Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    #memory input_key='input'可以避免讀到其他輸入
    #https://github.com/hwchase17/langchain/issues/1774
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", 
                                            input_key="input", 
                                            output_key='output', return_messages=True)

    my_agent = initialize_agent(
        tools=customize_tools,
        llm=global_llm_chat,
        agent='conversational-react-description',
        verbose=True,
        memory=memory,
        max_iterations=5,
        early_stopping_method='generate'
    )

    # 原始的prompt
    print('''input_variables=['input', 'chat_history', 'agent_scratchpad'] output_parser=None partial_variables={} template='Assistant is a large language model trained by OpenAI.\n\nAssistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n> SummarizeWebInformation: When you need to summarize web information after 2022, input should be key word\n> 查詢富邦銀行相關資訊: Useful for questions related to all bank related topics to get more information,         your action input here must be a single sentence query that correspond to the question\n> 查询复歌科技公司产品名称: 通过产品名称找到复歌科技产品描述时用的工具，输入应该是产品名称\n> 复歌科技公司相关信息: 当用户询问复歌科技公司相关的问题, 可以通过这个工具了解相关信息\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [SummarizeWebInformation, 查詢富邦銀行相關資訊, 查询复歌科技公司产品名称, 复歌科技公司相关信息]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nAI: [your response here]\n```\n\nBegin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}' template_format='f-string' validate_template=True''')

    #https://www.youtube.com/watch?v=q-HNphrWsDE
    agent_prompt_prefix = """
    Assistant is a large language model in 富邦銀行. Always answer question with Chinese, Write in a Persuasive, Descriptive style.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 

    It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
    Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 

    Unfortunately, assistant is terrible at current event or bank related topic, no matter how simple, assistant always refers to it's trusty tools for help and NEVER try to answer the question itself

    TOOLS:
    ------

    Assistant has access to the following tools:
    """
    agent_prompt_format_instructions = """To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No
    {ai_prefix}: [your response here]
    ```"""

    agent_prompt_suffix = """Begin!

    Previous conversation history:
    {chat_history}

    New user input: {input}
    {agent_scratchpad}"""

    #自己填充的prompt
    new_sys_msg = my_agent.agent.create_prompt(
        tools = customize_tools,
        prefix = agent_prompt_prefix,
        format_instructions= agent_prompt_format_instructions,
        suffix = agent_prompt_suffix,
        ai_prefix = "AI",
        human_prefix = "Human"
    ) #input_variables: Optional[List[str]] = None
    my_agent.agent.llm_chain.prompt = new_sys_msg
    my_agent.agent.llm_chain.prompt.output_parser = CustomOutputParser()
    my_agent.agent.llm_chain.prompt
    
    return my_agent

"""
my_agent.run('我叫吳花油')

my_agent.run('我的名字是什麼')

my_agent.run('幫我翻譯這句話成英文：您好，請問何時能夠洽談合作')

my_agent.run('Expected directory, got file: 在python遇到')

my_agent.run('''I have a dataframe, the three columns named 'played_duration', title_id, user_id, I want to know which title_id is the most popular. please add played_duration by title_id and return the title and their sum list''')

my_agent.run('''I want to sort them by their sum, the largest at front, return a list of title_id''')

my_agent.run('我昨天弄丟信用卡了，幫我搜尋補發方法')

"""


