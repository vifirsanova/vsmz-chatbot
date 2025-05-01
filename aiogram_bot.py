import asyncio
import json
import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

class FeedbackStates(StatesGroup):
    initial = State()
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

class TimeoutManager:
    def __init__(self):
        self.timers = {}
    
    async def reset(self, chat_id: int):
        if chat_id in self.timers:
            self.timers[chat_id].cancel()
            del self.timers[chat_id]
    
    async def set(self, chat_id: int, state: FSMContext, timeout: int = 300):
        await self.reset(chat_id)
        
        async def timeout_callback():
            await asyncio.sleep(timeout)  # Переносим sleep в начало callback
            try:
                current_state = await state.get_state()
                if current_state is not None:
                    await bot.send_message(
                        chat_id,
                        "Спасибо за уделенное время! Будем рады видеть вас снова!",
                        reply_markup=types.ReplyKeyboardRemove()
                    )
                    await state.clear()
            except Exception as e:
                logging.error(f"Timeout error: {e}")
            finally:
                if chat_id in self.timers:
                    del self.timers[chat_id]
        
        self.timers[chat_id] = asyncio.create_task(timeout_callback())

timeout_manager = TimeoutManager()

def save_to_json(data: dict):
    try:
        with open('feedback.json', 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        logging.error(f"Error saving data: {e}")

@dp.message(F.text == "/start")
async def start_feedback(message: types.Message, state: FSMContext):
    await timeout_manager.reset(message.chat.id)
    await state.clear()
    
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Начать опрос"))

    await message.answer(
        "Здравствуйте! Спасибо за посещение музея. Нажмите кнопку, чтобы начать опрос.",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )
    await state.set_state(FeedbackStates.initial)
    await timeout_manager.set(message.chat.id, state)

@dp.message(F.text == "Начать опрос", FeedbackStates.initial)
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
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.initial)
async def handle_initial_random(message: types.Message, state: FSMContext):
    await timeout_manager.set(message.chat.id, state)
    await message.answer("Пожалуйста, нажмите 'Начать опрос'")

@dp.message(FeedbackStates.gender, F.text.in_(["Мужской", "Женский", "Предпочитаю не указывать"]))
async def process_gender(message: types.Message, state: FSMContext):
    await state.update_data(gender=message.text)
    await message.answer(
        "Укажите ваш возраст:",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.set_state(FeedbackStates.age)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.gender)
async def wrong_gender(message: types.Message, state: FSMContext):
    await message.answer("Пожалуйста, выберите вариант из кнопок ниже")
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.age)
async def process_age(message: types.Message, state: FSMContext):
    if not message.text.isdigit():
        await message.answer("Пожалуйста, введите возраст числом:")
        await timeout_manager.set(message.chat.id, state)
        return

    age = int(message.text)
    if not 1 <= age <= 120:
        await message.answer("Введите реальный возраст (1-120):")
        await timeout_manager.set(message.chat.id, state)
        return

    await state.update_data(age=age)
    await message.answer("Из какого вы города?")
    await state.set_state(FeedbackStates.home_city)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.home_city)
async def process_home_city(message: types.Message, state: FSMContext):
    if len(message.text) < 2:
        await message.answer("Название города слишком короткое")
        await timeout_manager.set(message.chat.id, state)
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
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.visited_city)
async def process_visited_city(message: types.Message, state: FSMContext):
    await state.update_data(visited_city=message.text)
    await message.answer(
        "Что именно вы посетили? (экспозицию/выставку/экскурсию/мероприятие)",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.set_state(FeedbackStates.visited_events)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.visited_events)
async def process_visited_events(message: types.Message, state: FSMContext):
    if len(message.text) < 5:
        await message.answer("Пожалуйста, напишите более подробно")
        await timeout_manager.set(message.chat.id, state)
        return

    await state.update_data(visited_events=message.text)
    await message.answer("Что вам понравилось больше всего?")
    await state.set_state(FeedbackStates.liked)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.liked)
async def process_liked(message: types.Message, state: FSMContext):
    if len(message.text) < 5:
        await message.answer("Пожалуйста, напишите более развернуто")
        await timeout_manager.set(message.chat.id, state)
        return

    await state.update_data(liked=message.text)
    await message.answer("Что вам не понравилось или что можно улучшить?")
    await state.set_state(FeedbackStates.disliked)
    await timeout_manager.set(message.chat.id, state)

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
    await timeout_manager.reset(message.chat.id)

async def cleanup_timers():
    """Очистка завершенных таймеров"""
    while True:
        await asyncio.sleep(60)
        for chat_id in list(timeout_manager.timers.keys()):
            if timeout_manager.timers[chat_id].done():
                del timeout_manager.timers[chat_id]

async def main():
    while True:
        try:
            logging.info("Запуск бота...")
            await dp.start_polling(bot)
        except Exception as e:
            logging.error(f"Ошибка: {e}. Перезапуск через 10 секунд...")
            await asyncio.sleep(10)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())