import streamlit as st
import httpx
import asyncio
from streamlit_cookies_controller import CookieController
import os




BASE_URL = "http://" + os.environ.get("MAIN_API_URL", "localhost") + ":8000"

print("Main service", BASE_URL)
controller = CookieController()
MOVIES_PER_PAGE = 10
default_movies = [
    {'title': 'Toy Story (1995)', 'id': '1', 'genres': 'Adventure|Animation|Children|Comedy|Fantasy'},
    {'title': 'Jumanji (1995)', 'id': '2', 'genres': 'Adventure|Children|Fantasy'},
    {'title': 'Grumpier Old Men (1995)', 'id': '3', 'genres': 'Comedy|Romance'},
    {'title': 'Waiting to Exhale (1995)', 'id': '4', 'genres': 'Comedy|Drama|Romance'},
    {'title': 'Father of the Bride Part II (1995)', 'id': '5', 'genres': 'Comedy'},
    {'title': 'Heat (1995)', 'id': '6', 'genres': 'Action|Crime|Thriller'},
    {'title': 'Sabrina (1995)', 'id': '7', 'genres': 'Comedy|Romance'},
    {'title': 'Tom and Huck (1995)', 'id': '8', 'genres': 'Adventure|Children'},
]


def get_movies(page: int):
    """
    Запрашивает фильмы с бэкенда для указанной страницы.
    """
    headers = {
        "Authorization": f'Bearer {get_cookie("users_access_token")}'
    }

    offset = (page - 1) * MOVIES_PER_PAGE
    limit = MOVIES_PER_PAGE
    url = f"{BASE_URL}/movies/?offset={offset}&limit={limit}"
    response = httpx.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.warning(f"Зарегистрируйтесь, либо войдите в уже созданный аккаунт!")
        st.stop()


async def display_movies(movies):
    """
    Отображает список фильмов.
    """
    for movie in movies:
        st.write(f"**{movie['title']}**")
        st.write(f"Жанры: {movie['genres']}")
        if st.button(f"Смотреть: {movie['title']}"):
            # Формируем URL для получения видео
            video_url = f"{BASE_URL}/streaming/get/{movie['id']}"

            headers = {
                "Authorization": f'Bearer {get_cookie("users_access_token")}'
            }

            # Проверяем доступность видео
            try:
                async with httpx.AsyncClient() as client:
                    # Получаем видео с токеном авторизации
                    response = await client.get(video_url, headers=headers)
                    if response.status_code == 200:
                        # Воспроизводим видео
                        st.video(response.content)
                    else:
                        st.error(f"Не удалось загрузить видео. Код ошибки: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка при загрузке видео: {e}")
        st.write("---")




def centeredHeader(word):
    st.markdown(
        f"""
        <h1 style='text-align: center;'>{word}</h1>
        """,
        unsafe_allow_html=True
    )
    st.divider()

def get_cookie(key):
    return controller.get(key)

async def watching_movies():
    centeredHeader("Просмотр фильмов")
    # Инициализация состояния для хранения выбранного фильма
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = None

    if st.button("Получить список рекомендованных фильмов"):
        url = f"{BASE_URL}/movies/recs/"
        headers = {
            "Authorization": f'Bearer {get_cookie("users_access_token")}'
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                list_movie = response.json()
                if not list_movie:
                    list_movie = default_movies
                # Сохраняем список фильмов в сессионное состояние
                st.session_state.move_options = {f"{move['title']} - Жанры:({move['genres']})": move['id'] for move in list_movie}
            else:
                st.warning(f"Зарегистрируйтесь, либо войдите в уже созданный аккаунт!")
                st.stop()

    # Отображаем список фильмов, если он есть
    if "move_options" in st.session_state:
        st.write("Выберите фильм для просмотра:")
        selected_movie = st.selectbox("Фильмы", list(st.session_state.move_options.keys()))

        # Обновляем выбранный фильм только после нажатия кнопки
        if st.button("Выбрать фильм"):
            st.session_state.selected_movie = selected_movie

    if st.session_state.selected_movie:
        st.write(f"Вы выбрали фильм: {st.session_state.selected_movie}")

        # Получаем ID выбранного фильма
        movie_id = st.session_state.move_options[st.session_state.selected_movie]

        # Формируем URL для получения видео
        video_url = f"{BASE_URL}/streaming/get/{movie_id}"

        headers = {
                "Authorization": f'Bearer {get_cookie("users_access_token")}'
        }

        # Проверяем доступность видео
        try:
            async with httpx.AsyncClient() as client:
                # Получаем видео с токеном авторизации
                response = await client.get(video_url, headers=headers)
                if response.status_code == 200:
                    # Воспроизводим видео
                    st.video(response.content)
                else:
                    st.error(f"Не удалось загрузить видео. Код ошибки: {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка при загрузке видео: {e}")

    st.divider()
    st.header("Все фильмы:")
    st.divider()
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

        # Получаем фильмы для текущей страницы
    movies = get_movies(st.session_state.current_page)

    # Отображаем фильмы
    if movies:
        await display_movies(movies)
    else:
        st.write("Фильмы не найдены.")

    # Кнопки для перехода между страницами
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Предыдущая страница") and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()
    with col2:
        st.write(f"Страница {st.session_state.current_page}")
    with col3:
        if st.button("Следующая страница"):
            st.session_state.current_page += 1
            st.rerun()




async def registration():
    centeredHeader("Регистрация")
    user_name = st.text_input("Имя пользователя")
    full_name = st.text_input("ФИО")
    password = st.text_input("Пароль", type="password")

    if st.button("Зарегистрироваться"):
        url = f"{BASE_URL}/register"
        data = {
            "username": user_name,
            "full_name": full_name,
            "password": password,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data)
            if response.status_code == 200:
                st.success("Registration was successful!")
            else:
                st.error(f"Error: {response.status_code}, {response.text}")

async def login():
    centeredHeader("Вход")
    username = st.text_input("Имя пользователя")
    password = st.text_input("Пароль", type="password")
    if st.button("Войти"):
        url = f"{BASE_URL}/login"
        data = {
            "username": username,
            "password": password
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            if response.status_code == 200:
                access_token = response.json().get("access_token")
                if access_token:
                    controller.set("users_access_token", access_token)
                    st.success("Вы успешно вошли в систему!")
                else:
                    st.error("The token was not found in the server response!")
            else:
                st.error(f"Error: {response.status_code}, {response.text}")

choice = st.sidebar.radio("Выберите действие", ["Регистрация", "Вход", "Просмотр фильмов"])
if choice == "Регистрация":
    asyncio.run(registration())
elif choice == "Вход":
    asyncio.run(login())
elif choice == "Просмотр фильмов":
    asyncio.run(watching_movies())