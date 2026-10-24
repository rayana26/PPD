import base64
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY отсутствует в переменных окружения (.env)")

app = FastAPI()

llm = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    temperature=0
)

class SignSchema(BaseModel):
    class_id: int
    class_name: str
    category: str
    description: str

SYSTEM_PROMPT = """
Ты — опытный эксперт по ПДД. Определи дорожный знак на фотографии.
Заполни структуру JSON:
- class_id: числовой ID знака (придумай логичный индекс)
- class_name: Название знака на английском
- category: Категория знака на русском (например: Запрещающий знак)
- description: Описание ограничения или правила на русском

Если на фотографии вообще нет дорожного знака, верни class_id равный -1.
"""

structured_llm = llm.with_structured_output(SignSchema, method="json_schema")


@app.post('/pdd/predict/')
async def check_image_pdd(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='Файл не был загружен')

        image_base64 = base64.b64encode(image_data).decode('utf-8')

        message = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {'type': 'text', 'text': 'Какой дорожный знак на фото?'},
                {
                    'type': 'image_url',
                    'image_url': f'data:{file.content_type};base64,{image_base64}'
                },
            ])
        ]

        result = await structured_llm.ainvoke(message)

        if not result or result.class_id == -1:
            raise HTTPException(status_code=422, detail='Дорожный знак не найден')

        return result.model_dump()

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Ошибка на сервере: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8004)
