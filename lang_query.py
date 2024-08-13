from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain



def lang_response(user_query,attendance_data):

    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct" #"distilbert/distilbert-base-cased-distilled-squad"  
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128,temperature =0.7,stop_sequences=['User query'])

    # LangChain Setup
    template = """You are a helpful assistant. The user will ask about attendance records.Provide a concise answer based on the available data.

    Attendance data:
    {attendance_data}

    User query: {user_query}

    Provide a concise answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["attendance_data", "user_query"])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(attendance_data=attendance_data, user_query=user_query)
    return response
