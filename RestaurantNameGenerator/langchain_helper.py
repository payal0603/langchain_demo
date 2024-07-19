from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os
os.environ['OPENAI_API_KEY'] = ''

llm = OpenAI(temperature=0.7)

def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_tmpl_name = PromptTemplate(
        input_variables = ['cuisine'],
        template = "I want to open an restaurant for {cuisine} food. Suggest a fancy name for this."
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_tmpl_name, output_key="restaurant_name")

    # Chain 2: Menu Items
    prompt_tmpl_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template = "Suggest me a fancy name menu items for {restaurant_name}. Return it as comma separated list"
    )

    food_items_chain = LLMChain(llm=OpenAI(temperature=0.6), prompt=prompt_tmpl_items, output_key="menu_items")

    chain = SequentialChain(
        chains = [name_chain, food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', 'menu_items'],
    )

    response = chain({'cuisine': cuisine})
    return response

if __name__ == '__main__':
    print(generate_restaurant_name_and_items('Italian'))
