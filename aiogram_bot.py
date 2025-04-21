import asyncio
import json
import aiogram
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

class FeedbackStates(StatesGroup):
    gender = State()
    age = State()
    home_city = State()
    visited_city = State()
    visited_events = State()
    liked = State()
    disliked = State()

load_dotenv()
bot_token = os.getenv("BOT_TOKEN")

bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

def save_to_json(data: dict):
    try:
        with open('feedback.json', 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"Error saving data: {e}")

@dp.message(F.text == "/start")
async def start_feedback(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Начать опрос"))

    await message.answer(
        "Здравствуйте! Спасибо за посещение музея. Нажмите кнопку, чтобы начать опрос.",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

@dp.message(F.text == "Начать опрос")
async def start_survey(message: types.Message, state: FSMContext):
    builder = ReplyKeyboardBuilder()
    for gender in ["Мужской", "Женский", "Предпочитаю не указывать"]:
        builder.add(types.KeyboardButton(text=gender))
    builder.adjust(2)

    await message.answer(
        "Укажите ваш пол:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )
    await state.set_state(FeedbackStates.gender)

@dp.message(FeedbackStates.gender)
async def process_gender(message: types.Message, state: FSMContext):
    if message.text not in ["Мужской", "Женский", "Предпочитаю не указывать"]:
        await message.answer("Пожалуйста, выберите вариант из кнопок ниже")
        return

    await state.update_data(gender=message.text)
    await message.answer(
        "Укажите ваш возраст:",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.set_state(FeedbackStates.age)

@dp.message(FeedbackStates.age)
async def process_age(message: types.Message, state: FSMContext):
    if not message.text.isdigit():
        await message.answer("Пожалуйста, введите возраст числом:")
        return

    age = int(message.text)
    if not 1 <= age <= 120:
        await message.answer("Введите реальный возраст (1-120):")
        return

    await state.update_data(age=age)
    await message.answer("Из какого вы города?")
    await state.set_state(FeedbackStates.home_city)

@dp.message(FeedbackStates.home_city)
async def process_home_city(message: types.Message, state: FSMContext):
    if len(message.text) < 2:
        await message.answer("Название города слишком короткое")
        return

    await state.update_data(home_city=message.text)

    builder = ReplyKeyboardBuilder()
    cities = ["Владимир", "Суздаль", "Гусь-Хрустальный",
             "с. Муромцево", "пос. Боголюбово", "Юрьев-Польский"]
    for city in cities:
        builder.add(types.KeyboardButton(text=city))
    builder.adjust(2)
    builder.add(types.KeyboardButton(text="Другое"))

    await message.answer(
        "Какой город вы посетили?",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )
    await state.set_state(FeedbackStates.visited_city)

@dp.message(FeedbackStates.visited_city)
async def process_visited_city(message: types.Message, state: FSMContext):
    await state.update_data(visited_city=message.text)
    await message.answer(
        "Что именно вы посетили? (экспозицию/выставку/экскурсию/мероприятие)",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.set_state(FeedbackStates.visited_events)

@dp.message(FeedbackStates.visited_events)
async def process_visited_events(message: types.Message, state: FSMContext):
    if len(message.text) < 5:
        await message.answer("Пожалуйста, напишите более подробно")
        return

    await state.update_data(visited_events=message.text)
    await message.answer("Что вам понравилось больше всего?")
    await state.set_state(FeedbackStates.liked)

@dp.message(FeedbackStates.liked)
async def process_liked(message: types.Message, state: FSMContext):
    if len(message.text) < 5:
        await message.answer("Пожалуйста, напишите более развернуто")
        return

    await state.update_data(liked=message.text)
    await message.answer("Что вам не понравилось или что можно улучшить?")
    await state.set_state(FeedbackStates.disliked)

@dp.message(FeedbackStates.disliked)
async def process_disliked(message: types.Message, state: FSMContext):
    await state.update_data(disliked=message.text)
    user_data = await state.get_data()

    save_to_json(user_data)

    await message.answer(
        "Спасибо за обратную связь! Мы учтем ваши замечания.",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.clear()

# ================= ЗАПУСК БОТА =================
# Запуск бота
async def main():
    await dp.start_polling(bot)

# Запуск бота в зависимости от среды
if __name__ == '__main__':
    import asyncio

    try:
        # Проверяем, есть ли уже запущенный цикл событий
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если цикл уже запущен, используем create_task
            loop.create_task(main())
        else:
            # Если цикла нет, используем asyncio.run
            asyncio.run(main())
    except RuntimeError as e:
        # Обработка ошибок, если что-то пошло не так
        logging.error(f"Ошибка при запуске бота: {e}")
