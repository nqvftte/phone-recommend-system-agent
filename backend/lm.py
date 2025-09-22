import asyncio
import nest_asyncio
nest_asyncio.apply()
from agent3 import Smartphone_Recommendation_Agent, llm

async def recommend_smartphone(user_query):
    # Agent chạy async
    english_response = await Smartphone_Recommendation_Agent.run(user_query)

    # Dịch sang tiếng Việt
    translation_prompt = f"Translate the following text into natural Vietnamese:\n\n{english_response}"
    vietnamese_answer = llm.complete(translation_prompt).text

    return vietnamese_answer

async def main():
    question = " Điện thoại Xiaomi Redmi Note 14 5G có mấy màu?"
    answer = await recommend_smartphone(question)
    print("Kết quả gợi ý:\n", answer)

if __name__ == "__main__":
    asyncio.run(main())
