from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def create_prompt():
    template = """
    Ты умный AI bot. Ты собираешь корзину по текстовому сообщению.
    Предложи наиболее подходящий продукт.

    Ответ должен быть строго только в таком формате, не иначе:
    
    Ваш заказ:
    - сок "Добрый"  - 1 х 150 рублей
    - пицца "Пепперони" (25 см)  2 х 350 рублей
    Итого: 850 рублей

    
    Опирайся на этот контекст при ответе (формат контекста вопрос-ответ):
    {context}

    Question: {question}

    Ответ напиши на русском языке. Формируй только один заказ, не повторяйся. Не пиши пояснения и примечания.
    """
    return ChatPromptTemplate.from_template(template)

def create_chain(retriever, prompt, llm):

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
