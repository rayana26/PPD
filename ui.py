import requests
import streamlit as st

API_URL = "http://localhost:8004"

file = st.file_uploader("Upload image", type=["png", "jpeg", "jpg"])

if file is None:
    st.warning("Файл не выбран")
else:
    st.image(file, caption="Загруженное изображение")
    if st.button("Проверить"):
        try:
            with st.spinner("Распознаю..."):
                files = {"file": (file.name, file.getvalue(), file.type)}
                r = requests.post(f"{API_URL}/pdd/predict/", files=files, timeout=60)
            if r.ok:
                data = r.json()
                st.success(f"{data['class_name']} — {data['description']}")
                st.json(data)
            else:
                st.error(r.json()["detail"])
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
