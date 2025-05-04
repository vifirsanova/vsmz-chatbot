import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Set, Dict
from functools import lru_cache
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
import csv
import re
import json
from pymorphy3 import MorphAnalyzer
from transliterate import translit, detect_language
from pathlib import Path

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO)

# Инициализация анализатора
morph = MorphAnalyzer()

# Словарь для перевода кириллицы в глаголицу
GLAGOLITIC_MAP = {
    'а': 'Ⰰ', 'б': 'Ⰱ', 'в': 'Ⰲ', 'г': 'Ⰳ', 'д': 'Ⰴ',
    'е': 'Ⰵ', 'ё': 'Ⰵ', 'ж': 'Ⰶ', 'з': 'Ⰷ', 'и': 'Ⰻ', 'й': 'Ⰼ',
    'к': 'Ⰽ', 'л': 'Ⰾ', 'м': 'Ⰿ', 'н': 'Ⱀ', 'о': 'Ⱁ',
    'п': 'Ⱂ', 'р': 'Ⱃ', 'с': 'Ⱄ', 'т': 'Ⱅ', 'у': 'Ⱆ',
    'ф': 'Ⱇ', 'х': 'Ⱈ', 'ц': 'Ⱌ', 'ч': 'Ⱍ', 'ш': 'Ⱎ',
    'щ': 'Ⱋ', 'ъ': 'Ⱏ', 'ы': 'Ⰺ', 'ь': 'Ⱐ', 'ѣ': 'Ⱑ',
    'э': 'Ⰵ', 'ю': 'Ⱓ', 'я': 'Ⱔ',
    'А': 'Ⰰ', 'Б': 'Ⰱ', 'В': 'Ⰲ', 'Г': 'Ⰳ', 'Д': 'Ⰴ',
    'Е': 'Ⰵ', 'Ё': 'Ⰵ', 'Ж': 'Ⰶ', 'З': 'Ⰷ', 'И': 'Ⰻ', 'Й': 'Ⰼ',
    'К': 'Ⰽ', 'Л': 'Ⰾ', 'М': 'Ⰿ', 'Н': 'Ⱀ', 'О': 'Ⱁ',
    'П': 'Ⱂ', 'Р': 'Ⱃ', 'С': 'Ⱄ', 'Т': 'Ⱅ', 'У': 'Ⱆ',
    'Ф': 'Ⱇ', 'Х': 'Ⱈ', 'Ц': 'Ⱌ', 'Ч': 'Ⱍ', 'Ш': 'Ⱎ',
    'Щ': 'Ⱋ', 'Ъ': 'Ⱏ', 'Ы': 'Ⰺ', 'Ь': 'Ⱐ', 'Ѣ': 'Ⱑ',
    'Э': 'Ⰵ', 'Ю': 'Ⱓ', 'Я': 'Ⱔ'
}

def translate_to_glagolitic(text: str) -> str:
    """Переводит кириллический текст в глаголицу"""
    result = []
    for char in text:
        if char in GLAGOLITIC_MAP:
            result.append(GLAGOLITIC_MAP[char])
        else:
            result.append(char)  # Оставляем эмодзи и другие символы как есть
    return ''.join(result)

# Кэшируем результаты лемматизации
@lru_cache(maxsize=5000)
def normalize_word(word: str) -> str:
    """Приводит слово к нормальной форме (лемме) с кэшированием"""
    try:
        parsed = morph.parse(word)[0]
        return parsed.normal_form
    except:
        return word.lower()

def clean_word(word: str) -> str:
    """Удаляет повторяющиеся символы (например, 'кууурва' -> 'курва')"""
    return re.sub(r'(.)\1+', r'\1', word.lower())

def is_kurva_variant(word: str) -> bool:
    """Проверяет различные написания слова 'курва' с учетом транслитерации"""
    word_lower = word.lower()
    cleaned = clean_word(word_lower)
    
    # Основные варианты написания
    variants = {
        'курва', 'kurwa', 'kurva', 'kypва', 'kypwa', 
        'кура', 'куря', 'куре', 'куро', 'куру'
    }
    
    # Проверка прямого совпадения
    if word_lower in variants or cleaned in variants:
        return True
    
    # Проверка нормализованной формы
    normalized = normalize_word(cleaned)
    if normalized == 'курва':
        return True

    # Проверка транслитерированных вариантов
    try:
        if detect_language(word_lower) != 'ru':
            ru_word = translit(word_lower, 'ru')
            return ru_word in variants or normalize_word(ru_word) == 'курва'
    except:
        pass
    
    return False

def load_bad_words(filename: str = "bad_words.txt") -> set:
    """Загружает и нормализует список недопустимых слов"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            words = set()
            for line in file:
                word = line.strip().lower()
                if word:
                    words.add(word)
                    words.add(normalize_word(word))
            return words
    except FileNotFoundError:
        logging.warning(f"Файл {filename} не найден. Используется пустой список.")
        return set()
    except Exception as e:
        logging.error(f"Ошибка загрузки файла: {e}")
        return set()

# Загружаем словарь при старте
MAT_WORDS = load_bad_words()
MAT_RESPONSES = [
    "Пожалуйста, будьте вежливы. Давайте общаться культурно.",
    "У нас принято выражаться вежливо. Давайте без грубостей.",
    "Такие выражения недопустимы. Пожалуйста, следите за речью."
]

def load_city_dictionary(filename: str = "output_names.json") -> Set[str]:
    """Загружает и нормализует словарь городов из JSON-файла."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            cities = set()
            for city in data["names"]:
                # Нормализуем каждое название и его части (для составных названий)
                parts = re.split(r'[-–\s]', city)  # Разбиваем по дефисам и пробелам
                for part in parts:
                    if part:  # Игнорируем пустые строки
                        normalized = normalize_word(part.lower())
                        cities.add(normalized)
                # Добавляем полное название (для "Нью-Йорк" → "нью-йорк")
                full_normalized = normalize_word(city.lower().replace(' ', '-'))
                cities.add(full_normalized)
            return cities
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.error(f"Ошибка загрузки словаря городов: {e}")
        return set()

# Инициализация при старте бота
WORLD_CITIES = load_city_dictionary()

# Настройка базы данных
Base = declarative_base()

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default='in_progress')  # 'in_progress', 'completed', 'abandoned'
    gender = Column(String)
    age_group = Column(String)
    home_city = Column(String)
    visited_city = Column(String)
    visited_events = Column(Text)
    liked = Column(Text)
    disliked = Column(Text)

engine = create_engine('sqlite:///feedback.db', echo=False)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

class FeedbackStates(StatesGroup):
    initial = State()
    gender = State()
    age_group = State()
    home_city = State()
    visited_city = State()
    visited_events = State()
    liked = State()
    disliked = State()

class TranslationState(StatesGroup):
    waiting_for_text = State()

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
            await asyncio.sleep(timeout)
            try:
                current_state = await state.get_state()
                if current_state is not None:
                    user_data = await state.get_data()
                    feedback_id = user_data.get('feedback_id')
                    
                    if feedback_id:
                        # Помечаем запись как abandoned
                        with Session() as session:
                            feedback = session.query(Feedback).get(feedback_id)
                            if feedback and feedback.status == 'in_progress':
                                feedback.status = 'abandoned'
                                session.commit()
                    
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

class DatabaseManager:
    def __init__(self):
        self.session_factory = Session
    
    async def create_feedback(self) -> int:
        """Создает новую запись и возвращает её ID"""
        session = self.session_factory()
        try:
            new_feedback = Feedback()
            session.add(new_feedback)
            session.commit()
            return new_feedback.id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error in create_feedback: {e}")
            raise
        finally:
            session.close()

    async def get_current_question(self, feedback_id: int) -> str:
        """Определяет текущий вопрос на основе заполненных полей"""
        with Session() as session:
            feedback = session.query(Feedback).get(feedback_id)
            if not feedback:
                return "Продолжим опрос:"
            
            if feedback.gender is None:
                return "Укажите ваш пол:"
            elif feedback.age_group is None:
                return "Укажите вашу возрастную группу:"
            elif feedback.home_city is None:
                return "Из какого вы города?"
            elif feedback.visited_city is None:
                return "Какой город вы посетили?"
            elif feedback.visited_events is None:
                return "Что именно вы посетили? (экспозицию/выставку/экскурсию/мероприятие)"
            elif feedback.liked is None:
                return "Что вам понравилось больше всего?"
            elif feedback.disliked is None:
                return "Что вам не понравилось или что можно улучшить?"
            else:
                return "Продолжим опрос:"
    
    async def update_feedback(self, feedback_id: int, field: str, value: str | int) -> bool:
        """Обновляет указанное поле записи по ID"""
        session = self.session_factory()
        try:
            feedback = session.query(Feedback).get(feedback_id)
            if feedback:
                setattr(feedback, field, value)
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error in update_feedback: {e}")
            return False
        finally:
            session.close()

    async def purge_in_progress(self) -> int:
        """Удаляет осиротевшие записи в состоянии in_progress"""
        session = self.session_factory()
        try:
            result = session.query(Feedback)\
                .filter(Feedback.status == 'in_progress')\
                .delete(synchronize_session=False)
            session.commit()
            return result
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Ошибка удаления in_progress записей: {e}")
            return 0
        finally:
            session.close()

    async def purge_abandoned(self, days: int = 1, chunk_size: int = 100) -> int:
        """Удаляет abandoned записи пачками с защитой от блокировки БД"""
        session = self.session_factory()
        total_deleted = 0
        try:
            while True:
                cutoff = datetime.utcnow() - timedelta(days=days)
                
                # Получаем ID для пачки записей
                ids_to_delete = [
                    r[0] for r in session.query(Feedback.id)
                    .filter(
                        Feedback.status == 'abandoned',
                        Feedback.last_activity < cutoff
                    )
                    .limit(chunk_size)
                    .all()
                ]
                
                if not ids_to_delete:
                    break
                    
                # Удаляем по ID
                deleted = session.query(Feedback)\
                    .filter(Feedback.id.in_(ids_to_delete))\
                    .delete(synchronize_session=False)
                    
                session.commit()
                total_deleted += deleted
                logging.debug(f"Удалено {deleted} записей")
                
            return total_deleted
        
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Ошибка удаления: {e}")
            return 0
        finally:
            session.close()
    
    async def complete_feedback(self, feedback_id: int) -> bool:
        """Помечает опрос как завершенный"""
        session = self.session_factory()
        try:
            feedback = session.query(Feedback).get(feedback_id)
            if feedback:
                feedback.status = 'completed'
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error in complete_feedback: {e}")
            return False
        finally:
            session.close()
    
    async def cleanup_abandoned(self, hours: int = 1) -> int:
        """Помечает неактивные записи как abandoned"""
        session = self.session_factory()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            result = session.query(Feedback).filter(
                Feedback.status == 'in_progress',
                Feedback.last_activity < cutoff_time
            ).update({'status': 'abandoned'})
            session.commit()
            return result
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Ошибка пометки abandoned: {e}")
            return 0
        finally:
            session.close()
   
    async def export_to_csv(self, filename: str = "feedback.csv") -> bool:
        """Экспортирует завершенные опросы в CSV с обработкой home_city"""
        session = self.session_factory()
        try:
            feedbacks = session.query(Feedback).filter(
                Feedback.status == 'completed'
            ).all()
            
            if not feedbacks:
                return False
                
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'id', 'timestamp', 'gender', 'age_group', 
                    'home_city', 'visited_city', 
                    'visited_events', 'liked', 'disliked'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for feedback in feedbacks:
                    # Обрабатываем только home_city
                    home_city = get_nominative_city_name(feedback.home_city) if feedback.home_city else None
                    
                    writer.writerow({
                        'id': feedback.id,
                        'timestamp': feedback.timestamp.isoformat(),
                        'gender': feedback.gender,
                        'age_group': feedback.age_group,
                        'home_city': home_city,  # Обработанное название
                        'visited_city': feedback.visited_city,  # Оригинал как есть
                        'visited_events': feedback.visited_events,
                        'liked': feedback.liked,
                        'disliked': feedback.disliked
                    })
            return True
        except Exception as e:
            logging.error(f"Error exporting to CSV: {e}")
            return False
        finally:
            session.close()

db_manager = DatabaseManager()

async def check_mat_and_respond(message: types.Message, state: FSMContext) -> bool:
    """Проверяет сообщение на мат с учётом всех улучшений"""
    if not MAT_WORDS:
        return False

    # Очищаем текст и разбиваем на слова
    text = re.sub(r'[^\w\s]', '', message.text.lower())
    words = text.split()
    
    has_mat = False
    for word in words:
        if is_kurva_variant(word) or word in MAT_WORDS or normalize_word(word) in MAT_WORDS:
            has_mat = True
            break
    
    if not has_mat:
        return False

    user_data = await state.get_data()
    current_state = await state.get_state()
    
    # Если опрос ещё не начат (initial state) или в режиме перевода
    if current_state in [FeedbackStates.initial.state, TranslationState.waiting_for_text.state]:
        mat_count = user_data.get("mat_count", 0) + 1
        await state.update_data(mat_count=mat_count)
        
        if mat_count >= 3:
            await message.answer(
                "К сожалению, мы вынуждены прекратить общение. Всего доброго!",
                reply_markup=types.ReplyKeyboardRemove()
            )
            await state.clear()
            await timeout_manager.reset(message.chat.id)
            return True
        
        # Отправляем ответ на мат отдельным сообщением
        await message.answer(MAT_RESPONSES[mat_count - 1])
        
        # В зависимости от состояния отправляем соответствующее приглашение
        if current_state == FeedbackStates.initial.state:
            builder = ReplyKeyboardBuilder()
            builder.add(types.KeyboardButton(text="Начать опрос"))
            builder.add(types.KeyboardButton(text="Перевод на глаголицу"))
            await message.answer(
                "Пожалуйста, выберите действие:",
                reply_markup=builder.as_markup(resize_keyboard=True)
            )
        elif current_state == TranslationState.waiting_for_text.state:
            await message.answer(
                "Введите текст на кириллице для перевода в глаголицу:",
                reply_markup=types.ReplyKeyboardRemove()
            )
        return True
    
    # Если опрос уже начат (любое другое состояние)
    feedback_id = user_data.get("feedback_id")
    if not feedback_id:
        return False
        
    mat_count = user_data.get("mat_count", 0) + 1
    await state.update_data(mat_count=mat_count)
    
    if mat_count >= 3:
        await db_manager.update_feedback(feedback_id, "status", "abandoned")
        await message.answer(
            "К сожалению, мы вынуждены прекратить общение. Всего доброго!",
            reply_markup=types.ReplyKeyboardRemove()
        )
        await state.clear()
        await timeout_manager.reset(message.chat.id)
        return True
    
    try:
        current_question = await db_manager.get_current_question(feedback_id)
        
        # Отправляем ответ на мат отдельным сообщением
        await message.answer(MAT_RESPONSES[mat_count - 1])
        
        # Отправляем повтор вопроса отдельным сообщением
        current_state = await state.get_state()
        
        if current_state == FeedbackStates.gender.state:
            builder = ReplyKeyboardBuilder()
            for gender in ["Мужской", "Женский", "Предпочитаю не указывать"]:
                builder.add(types.KeyboardButton(text=gender))
            builder.adjust(2)
            await message.answer(
                f"Вернемся к вопросу:\n{current_question}",
                reply_markup=builder.as_markup(resize_keyboard=True)
            )
        
        elif current_state == FeedbackStates.visited_city.state:
            builder = ReplyKeyboardBuilder()
            cities = ["Владимир", "Суздаль", "Гусь-Хрустальный",
                     "с. Муромцево", "пос. Боголюбово", "Юрьев-Польский"]
            for city in cities:
                builder.add(types.KeyboardButton(text=city))
            builder.adjust(2)
            builder.add(types.KeyboardButton(text="Другое"))
            await message.answer(
                f"Вернемся к вопросу:\n{current_question}",
                reply_markup=builder.as_markup(resize_keyboard=True))
        
        else:
            await message.answer(f"Вернемся к вопросу:\n{current_question}")
        
        return True
    except Exception as e:
        logging.error(f"Ошибка при обработке мата: {e}")
        return False

@dp.message(F.text == "/start")
async def start_feedback(message: types.Message, state: FSMContext):
    await state.update_data(mat_count=0)
    if await check_mat_and_respond(message, state):
        return
    await timeout_manager.reset(message.chat.id)
    await state.clear()
    
    try:
        feedback_id = await db_manager.create_feedback()
        await state.update_data(feedback_id=feedback_id)
    except Exception as e:
        logging.error(f"Error creating feedback record: {e}")
        await message.answer("Произошла ошибка. Пожалуйста, попробуйте позже.")
        return
    
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Начать опрос"))
    builder.add(types.KeyboardButton(text="Перевод на глаголицу"))

    await message.answer(
        "Здравствуйте! Спасибо за посещение музея. Выберите действие:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )
    await state.set_state(FeedbackStates.initial)
    await timeout_manager.set(message.chat.id, state)

@dp.message(F.text == "Начать опрос", FeedbackStates.initial)
async def start_survey(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
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

@dp.message(F.text == "Перевод на глаголицу", FeedbackStates.initial)
async def start_glagolitic_translation(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    await state.set_state(TranslationState.waiting_for_text)
    await message.answer(
        "Введите текст на кириллице для перевода в глаголицу:",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await timeout_manager.set(message.chat.id, state)

@dp.message(F.text == "Перевести ещё", TranslationState.waiting_for_text)
async def translate_more(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    await state.set_state(TranslationState.waiting_for_text)
    await message.answer(
        "Введите текст на кириллице для перевода в глаголицу:",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await timeout_manager.set(message.chat.id, state)

@dp.message(F.text == "Перейти к опросу", TranslationState.waiting_for_text)
async def switch_to_survey(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    await start_survey(message, state)

@dp.message(TranslationState.waiting_for_text)
async def handle_glagolitic_translation(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return

        # Проверяем, что текст содержит хотя бы одну кириллическую букву
    if any(char in GLAGOLITIC_MAP for char in message.text):
        translated = translate_to_glagolitic(message.text)
        
        builder = ReplyKeyboardBuilder()
        builder.add(types.KeyboardButton(text="Перевести ещё"))
        builder.add(types.KeyboardButton(text="Перейти к опросу"))
        
        await message.answer(
            f"Перевод на глаголицу:\n\n{translated}",
            reply_markup=builder.as_markup(resize_keyboard=True)
        )
    else:
        await message.answer("Пожалуйста, введите текст, содержащий кириллические символы.")
    
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.initial)
async def handle_initial_random(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    await timeout_manager.set(message.chat.id, state)
    await message.answer("Пожалуйста, нажмите 'Начать опрос'")

@dp.message(FeedbackStates.gender, F.text.in_(["Мужской", "Женский", "Предпочитаю не указывать"]))
async def process_gender(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    user_data = await state.get_data()
    feedback_id = user_data.get("feedback_id")
    
    if not feedback_id:
        await message.answer("Ошибка сессии. Пожалуйста, начните опрос заново (/start).")
        return
    
    success = await db_manager.update_feedback(feedback_id, "gender", message.text)
    if not success:
        await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте позже.")
        return
    
    # Создаем клавиатуру с возрастными группами
    builder = ReplyKeyboardBuilder()
    for group in ["до 18", "19-25", "26-40", "41-59", "Старше 60"]:
        builder.add(types.KeyboardButton(text=group))
    builder.adjust(2)  # Группируем кнопки по 2 в ряд

    await message.answer(
        "Укажите вашу возрастную группу:",  # Уточняем, что нужна группа, а не точный возраст
        reply_markup=builder.as_markup(resize_keyboard=True)  # Показываем кнопки
    )
    await state.set_state(FeedbackStates.age_group)  # Переходим в age_group, а не age
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.gender)
async def wrong_gender(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    await message.answer("Пожалуйста, выберите вариант из кнопок ниже")
    await timeout_manager.set(message.chat.id, state)

def get_age_group(text: str) -> tuple[str | None, str | None]:
    """Определяет группу из текста. Возвращает (группа, ошибка)."""
    # Ищем целые числа, включая отрицательные
    numbers = re.findall(r'-?\d+', text)  # Изменено на -?\d+
    if not numbers:
        return None, None
    
    try:
        age = int(numbers[0])
    except ValueError:
        return None, None
    
    if age <= 0:
        return None, "Пожалуйста, укажите свой настоящий возраст"
    elif age > 120:
        return None, "Пожалуйста, укажите свой настоящий возраст"
    
    if age <= 18:
        return "до 18", None
    elif 19 <= age <= 25:
        return "19-25", None
    elif 26 <= age <= 40:
        return "26-40", None
    elif 41 <= age <= 59:
        return "41-59", None
    else:
        return "Старше 60", None

@dp.message(FeedbackStates.age_group)
async def process_age_group(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return

    user_data = await state.get_data()
    feedback_id = user_data.get("feedback_id")

    if not feedback_id:
        await message.answer("Ошибка сессии. Пожалуйста, начните заново (/start).")
        return

    # Создаем клавиатуру (на случай, если ввод некорректный)
    builder = ReplyKeyboardBuilder()
    for group in ["до 18", "19-25", "26-40", "41-59", "Старше 60"]:
        builder.add(types.KeyboardButton(text=group))
    builder.adjust(2)

    # Если пользователь выбрал кнопку
    if message.text in ["до 18", "19-25", "26-40", "41-59", "Старше 60"]:
        age_group = message.text
    else:
        # Пытаемся распознать возраст из текста ("мне 25" → "19-25")
        age_group, error_msg = get_age_group(message.text)

        if error_msg:  # Если ввели 0, 999 и т.д.
            await message.answer(error_msg, reply_markup=builder.as_markup())
            return

        if not age_group:  # Если не распознано
            await message.answer(
                "Пожалуйста, выберите возрастную группу из кнопок ниже:",
                reply_markup=builder.as_markup()
            )
            return

    # Сохраняем группу и переходим к следующему вопросу
    success = await db_manager.update_feedback(feedback_id, "age_group", age_group)
    if not success:
        await message.answer("Ошибка сохранения. Попробуйте позже.")
        return

    await message.answer(
        "Из какого вы города?",
        reply_markup=types.ReplyKeyboardRemove()  # Убираем кнопки для следующего вопроса
    )
    await state.set_state(FeedbackStates.home_city)
    await timeout_manager.set(message.chat.id, state)

STOP_WORDS = {"из", "в", "город", "приехал", "живу", "родом", "еду", "прибыл", "прибыла", "приехала"}
MIN_CITY_LENGTH = 2

def get_nominative_city_name(city_name: str) -> str:
    """Приводит город к именительному падежу, сохраняя стандартные правила написания."""
    if not city_name:
        return city_name

    # Разбиваем на слова и разделители (дефисы/пробелы)
    parts = re.split(r'([- ])', city_name)
    processed_parts = []

    for part in parts:
        if part in ('-', ' '):
            processed_parts.append(part)
            continue

        # Приводим слово к нормальной форме (лемме)
        try:
            parsed = morph.parse(part)[0]
            lemma = parsed.normal_form
        except:
            lemma = part.lower()

        # Правила регистра:
        # 1. Если слово было с заглавной (Петербург) → сохраняем capitalize
        # 2. Если слово было в верхнем регистре (ЙОРК) → capitalize (Йорк)
        # 3. Иначе → lower (предлоги, частицы)
        if part == part.upper():
            processed_part = lemma.capitalize()  # НЬЮ → Нью, ЙОРК → Йорк
        elif part[0].isupper():
            processed_part = lemma.capitalize()  # Петербург → Петербург
        else:
            processed_part = lemma.lower()  # москва → москва

        processed_parts.append(processed_part)

    # Собираем название обратно
    result = "".join(processed_parts)

    # Автоматически capitalize после дефиса (санкт-петербург → Санкт-Петербург)
    if '-' in result:
        result = re.sub(
            r'(^|[- ])([а-яёa-z])',
            lambda m: m.group(1) + m.group(2).upper(),
            result
        )

    return result

def extract_city_from_text(text: str) -> Optional[str]:
    """Извлекает город из текста с надежной обработкой"""
    logging.info(f"[extract_city_from_text] Начало обработки текста: '{text}'")
    
    # Удаляем лишние символы, сохраняя дефисы и пробелы
    cleaned = re.sub(r'[^\w\s-]', '', text)
    logging.info(f"[extract_city_from_text] Текст после очистки: '{cleaned}'")
    
    words = re.findall(r'[\w-]+', cleaned.lower())
    logging.info(f"[extract_city_from_text] Все слова: {words}")
    
    words = [w for w in words if w not in STOP_WORDS and len(w) >= MIN_CITY_LENGTH]
    logging.info(f"[extract_city_from_text] Отфильтрованные слова: {words}")
    
    # Проверяем варианты от самых длинных к коротким
    for word_count in range(min(3, len(words)), 0, -1):
        logging.info(f"[extract_city_from_text] Проверяем комбинации из {word_count} слов")
        
        for i in range(len(words) - word_count + 1):
            current_phrase = words[i:i+word_count]
            
            # Вариант с дефисами
            phrase_hyphen = '-'.join(current_phrase)
            normalized_hyphen = normalize_word(phrase_hyphen)
            logging.info(f"[extract_city_from_text] Проверка комбинации: '{phrase_hyphen}' -> нормализовано: '{normalized_hyphen}'")
            
            if normalized_hyphen in WORLD_CITIES:
                # Находим оригинальное написание в тексте
                match = re.search(re.escape(phrase_hyphen), cleaned, re.IGNORECASE)
                if match:
                    original = match.group()
                    logging.info(f"[extract_city_from_text] Найдено совпадение в словаре: '{original}'")
                    return get_nominative_city_name(original)
            
            # Вариант с пробелами
            phrase_space = ' '.join(current_phrase)
            normalized_space = normalize_word(phrase_space.replace(' ', '-'))
            logging.info(f"[extract_city_from_text] Проверка комбинации: '{phrase_space}' -> нормализовано: '{normalized_space}'")
            
            if normalized_space in WORLD_CITIES:
                match = re.search(re.escape(phrase_space), cleaned, re.IGNORECASE)
                if match:
                    original = match.group()
                    logging.info(f"[extract_city_from_text] Найдено совпадение в словаре: '{original}'")
                    return get_nominative_city_name(original)
    
    logging.warning("[extract_city_from_text] Не удалось найти город в тексте")
    return None

@dp.message(FeedbackStates.home_city)
async def process_home_city(message: types.Message, state: FSMContext):
    logging.info(f"\n[process_home_city] Начало обработки сообщения: '{message.text}'")
    
    if await check_mat_and_respond(message, state):
        logging.warning("[process_home_city] Обнаружен мат в сообщении")
        return

    input_text = message.text
    city = extract_city_from_text(input_text)
    
    # Принудительно капитализируем первую букву, даже если город был введен в нижнем регистре
    if city:
        city = city[0].upper() + city[1:]  # "москва" -> "Москва"
    
    logging.info(f"[process_home_city] Извлеченный город (после капитализации): '{city}'")
    
    if not city:
        logging.warning("[process_home_city] Город не распознан")
        await message.answer("Не удалось распознать город. Пожалуйста, укажите в формате: «Москва», «Санкт-Петербург»")
        await timeout_manager.set(message.chat.id, state)
        return

    # Дополнительная проверка перед сохранением
    normalized_city = normalize_word(city.lower().replace(' ', '-'))
    logging.info(f"[process_home_city] Нормализованная форма города: '{normalized_city}'")
    
    if normalized_city not in WORLD_CITIES:
        logging.error(f"[process_home_city] Город не найден в словаре: '{city}' (нормализованный: '{normalized_city}')")
        await message.answer("Указанный город не найден в списке. Пожалуйста, укажите другой.")
        return

    # Сохраняем в БД (уже с правильным регистром)
    user_data = await state.get_data()
    feedback_id = user_data.get("feedback_id")
    
    if not feedback_id:
        logging.error("[process_home_city] Не найден feedback_id")
        await message.answer("Ошибка сессии. Начните заново (/start).")
        return

    logging.info(f"[process_home_city] Сохраняем город '{city}' для feedback_id {feedback_id}")
    success = await db_manager.update_feedback(feedback_id, "home_city", city)
    
    if not success:
        logging.error("[process_home_city] Ошибка сохранения в БД")
        await message.answer("Ошибка сохранения. Попробуйте позже.")
        return
        
    logging.info("[process_home_city] Успешно сохранено, переходим к следующему вопросу")
    
    # Переход к следующему вопросу
    builder = ReplyKeyboardBuilder()
    for btn in ["Владимир", "Суздаль", "Гусь-Хрустальный", "с. Муромцево", "пос. Боголюбово", "Юрьев-Польский", "Другое"]:
        builder.add(types.KeyboardButton(text=btn))
    builder.adjust(2)
    
    await message.answer(
        "Какой город вы посетили?",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )
    await state.set_state(FeedbackStates.visited_city)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.visited_city, F.text.in_(["Владимир", "Суздаль", "Гусь-Хрустальный",
                                                   "с. Муромцево", "пос. Боголюбово", "Юрьев-Польский", "Другое"]))
async def process_visited_city(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    user_data = await state.get_data()
    feedback_id = user_data.get("feedback_id")
    
    if not feedback_id:
        await message.answer("Ошибка сессии. Пожалуйста, начните опрос заново (/start).")
        return
    
    success = await db_manager.update_feedback(feedback_id, "visited_city", message.text)
    if not success:
        await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте позже.")
        return
    
    await message.answer(
        "Что именно вы посетили? (экспозицию/выставку/экскурсию/мероприятие)",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.set_state(FeedbackStates.visited_events)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.visited_city)
async def wrong_visited_city(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    builder = ReplyKeyboardBuilder()
    cities = ["Владимир", "Суздаль", "Гусь-Хрустальный",
             "с. Муромцево", "пос. Боголюбово", "Юрьев-Польский"]
    for city in cities:
        builder.add(types.KeyboardButton(text=city))
    builder.adjust(2)
    builder.add(types.KeyboardButton(text="Другое"))
    
    await message.answer(
        "Пожалуйста, выберите город из списка:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.visited_events)
async def process_visited_events(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    user_data = await state.get_data()
    feedback_id = user_data.get("feedback_id")
    
    if not feedback_id:
        await message.answer("Ошибка сессии. Пожалуйста, начните опрос заново (/start).")
        return
    
    if len(message.text) < 5:
        await message.answer("Пожалуйста, напишите более подробно")
        await timeout_manager.set(message.chat.id, state)
        return

    success = await db_manager.update_feedback(feedback_id, "visited_events", message.text)
    if not success:
        await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте позже.")
        return
    
    await message.answer("Что вам понравилось больше всего?")
    await state.set_state(FeedbackStates.liked)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.liked)
async def process_liked(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    user_data = await state.get_data()
    feedback_id = user_data.get("feedback_id")
    
    if not feedback_id:
        await message.answer("Ошибка сессии. Пожалуйста, начните опрос заново (/start).")
        return
    
    if len(message.text) < 5:
        await message.answer("Пожалуйста, напишите более развернуто")
        await timeout_manager.set(message.chat.id, state)
        return

    success = await db_manager.update_feedback(feedback_id, "liked", message.text)
    if not success:
        await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте позже.")
        return
    
    await message.answer("Что вам не понравилось или что можно улучшить?")
    await state.set_state(FeedbackStates.disliked)
    await timeout_manager.set(message.chat.id, state)

@dp.message(FeedbackStates.disliked)
async def process_disliked(message: types.Message, state: FSMContext):
    if await check_mat_and_respond(message, state):
        return
    user_data = await state.get_data()
    feedback_id = user_data.get("feedback_id")
    
    if not feedback_id:
        await message.answer("Ошибка сессии. Пожалуйста, начните опрос заново (/start).")
        return
    
    success = await db_manager.update_feedback(feedback_id, "disliked", message.text)
    if not success:
        await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте позже.")
        return
    
    # Помечаем опрос как завершенный
    await db_manager.complete_feedback(feedback_id)
    
    # Экспорт в CSV
    await db_manager.export_to_csv()
    
    await message.answer(
        "Спасибо за обратную связь! Мы учтем ваши замечания.",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await state.clear()
    await timeout_manager.reset(message.chat.id)

async def periodic_cleanup():
    """Фоновая очистка: пометка и удаление записей"""
    while True:
        try:
            # 1. Помечаем неактивные записи
            marked = await db_manager.cleanup_abandoned(hours=1)
            if marked > 0:
                logging.info(f"Помечено как abandoned: {marked} записей")
            
            # 2. Удаляем старые abandoned
            purged = await db_manager.purge_abandoned(days=1)
            if purged > 0:
                logging.info(f"Удалено abandoned: {purged} записей")
                
        except Exception as e:
            logging.error(f"Ошибка в фоновой очистке: {e}")
        finally:
            await asyncio.sleep(3600)  # Интервал 1 час

async def log_database_state():
    """Логирует текущее состояние базы данных"""
    with Session() as session:
        records = session.query(Feedback).all()
        logging.info("\n=== ТЕКУЩЕЕ СОСТОЯНИЕ БАЗЫ ДАННЫХ ===")
        logging.info(f"Всего записей: {len(records)}")
        
        status_counts = {
            'in_progress': 0,
            'completed': 0,
            'abandoned': 0
        }
        
        for r in records:
            status_counts[r.status] += 1
            
        logging.info(f"Статусы записей: {status_counts}")
        
        # Логируем последние 5 записей для примера
        logging.info("Последние записи:")
        for r in records[-5:]:
            logging.info(
                f"ID: {r.id}, Статус: {r.status}, "
                f"Последняя активность: {r.last_activity}, "
                f"Данные: пол={r.gender or '-'}, возраст={r.age_group or '-'}"
            )
        logging.info("=== КОНЕЦ ОТЧЕТА ===")

async def main():
    # Стартовая очистка
    try:
        purged_in_progress = await db_manager.purge_in_progress()
        logging.info(f"Стартовая очистка: удалено in_progress записей: {purged_in_progress}")
        purged = await db_manager.purge_abandoned(days=0)
        logging.info(f"Стартовая очистка: удалено abandoned записей: {purged}")
    except Exception as e:
        logging.error(f"Ошибка стартовой очистки: {e}")
    # Логируем состояние БД при старте
    await log_database_state()
    # Запускаем фоновую задачу очистки
    asyncio.create_task(periodic_cleanup())
    
    try:
        logging.info("Запуск бота...")
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Ошибка: {e}")
    finally:
        # Логируем состояние БД при завершении
        await log_database_state()
        # При завершении работы экспортируем оставшиеся данные
        await db_manager.export_to_csv()

if __name__ == '__main__':
    asyncio.run(main())