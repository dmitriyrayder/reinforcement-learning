import streamlit as st
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="RL Система для Оптики", layout="wide")

# Заголовок
st.title("🤖 Reinforcement Learning: Оптимизация розничной сети")
st.markdown("---")

@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Загрузка и подготовка данных"""
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {str(e)}")
        st.stop()
    
    # Преобразование даты - убираем пробелы и парсим
    df['Datasales'] = df['Datasales'].astype(str).str.strip()
    df['Datasales'] = pd.to_datetime(df['Datasales'], format='%d.%m.%Y', errors='coerce')
    
    # Проверка на успешность парсинга
    invalid_dates = df['Datasales'].isna().sum()
    if invalid_dates > 0:
        st.warning(f"⚠️ Найдено {invalid_dates} записей с некорректными датами. Они будут исключены.")
        df = df.dropna(subset=['Datasales'])
    
    # Добавляем недостающие поля рандомно
    np.random.seed(42)
    
    # Себестоимость (60-80% от цены)
    df['Cost'] = df['Price'] * np.random.uniform(0.6, 0.8, len(df))
    df['Cost'] = df['Cost'].round(2)
    
    # Маржа
    df['Margin'] = df['Sum'] - (df['Cost'] * df['Qty'])
    
    # Уникальные магазины
    stores = df['Magazin'].unique()
    
    # Характеристики магазинов
    store_features = {}
    regions = ['Київ', 'Львів', 'Одеса', 'Харків', 'Дніпро']
    
    for store in stores:
        store_features[store] = {
            'region': np.random.choice(regions),
            'area_sqm': np.random.randint(50, 200),  # площадь магазина
            'traffic': np.random.randint(100, 500)  # средний трафик в день
        }
    
    df['Region'] = df['Magazin'].map(lambda x: store_features[x]['region'])
    df['Store_Area'] = df['Magazin'].map(lambda x: store_features[x]['area_sqm'])
    df['Daily_Traffic'] = df['Magazin'].map(lambda x: store_features[x]['traffic'])
    
    # Расчет остатков: +50% к среднему числу продаж по каждому товару в магазине
    sales_avg = df.groupby(['Magazin', 'Art'])['Qty'].mean().reset_index()
    sales_avg.columns = ['Magazin', 'Art', 'Avg_Sales']
    sales_avg['Stock'] = (sales_avg['Avg_Sales'] * 1.5).round(0).astype(int)
    
    df = df.merge(sales_avg[['Magazin', 'Art', 'Stock']], on=['Magazin', 'Art'], how='left')
    df['Stock'] = df['Stock'].fillna(5).astype(int)
    
    return df, store_features

class RetailEnvironment(gym.Env):
    """Среда для RL: управление распределением товара и маркетингом"""
    
    def __init__(self, df, stores, products, horizon_days=30):
        super(RetailEnvironment, self).__init__()
        
        self.df = df
        self.stores = stores
        self.products = products[:100]  # Ограничиваем для скорости
        self.horizon_days = horizon_days
        self.current_step = 0
        
        # Пространство действий: 
        # [магазин_индекс, товар_индекс, количество_для_перераспределения, промо_да/нет]
        self.action_space = spaces.MultiDiscrete([
            len(self.stores),  # выбор магазина
            len(self.products),  # выбор товара
            10,  # количество единиц товара (0-9)
            2   # промо акция (0=нет, 1=да)
        ])
        
        # Пространство состояний
        # [остатки_по_магазинам, продажи_за_неделю, маржа, день_месяца]
        self.observation_space = spaces.Box(
            low=0, high=1000, 
            shape=(len(self.stores) * len(self.products) + 10,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Сброс среды"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Инициализация остатков
        self.stocks = {}
        for store in self.stores:
            self.stocks[store] = {}
            for product in self.products:
                avg_stock = self.df[(self.df['Magazin'] == store) & 
                                   (self.df['Art'] == product)]['Stock'].mean()
                self.stocks[store][product] = int(avg_stock) if not np.isnan(avg_stock) else 5
        
        self.total_revenue = 0
        self.total_margin = 0
        self.actions_history = []
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Получение текущего состояния"""
        state = []
        
        # Остатки по магазинам (упрощенно - средние по топ продуктам)
        for store in self.stores[:5]:  # Берем первые 5 магазинов
            avg_stock = np.mean([self.stocks[store].get(p, 0) for p in self.products[:20]])
            state.append(avg_stock)
        
        # Дополнительные фичи
        state.extend([
            self.current_step / self.horizon_days,  # прогресс
            self.total_revenue / 100000,  # нормализованная выручка
            self.total_margin / 50000,  # нормализованная маржа
            len(self.actions_history) / 100  # количество действий
        ])
        
        # Дополняем до нужного размера
        while len(state) < self.observation_space.shape[0]:
            state.append(0)
        
        return np.array(state[:self.observation_space.shape[0]], dtype=np.float32)
    
    def step(self, action):
        """Выполнение действия"""
        store_idx, product_idx, qty, promo = action
        
        store = self.stores[store_idx]
        product = self.products[product_idx]
        
        # Проверяем наличие товара
        current_stock = self.stocks[store].get(product, 0)
        
        if current_stock <= 0:
            # Нет товара - отрицательная награда
            reward = -10
        else:
            # Симуляция продаж
            base_sales = min(qty + 1, current_stock)
            
            # Промо увеличивает продажи на 20-50%
            if promo == 1:
                sales_multiplier = np.random.uniform(1.2, 1.5)
                promo_cost = base_sales * 50  # стоимость промо
            else:
                sales_multiplier = 1.0
                promo_cost = 0
            
            actual_sales = int(base_sales * sales_multiplier)
            actual_sales = min(actual_sales, current_stock)
            
            # Получаем цену и себестоимость
            product_data = self.df[(self.df['Magazin'] == store) & 
                                   (self.df['Art'] == product)]
            
            if len(product_data) > 0:
                avg_price = product_data['Price'].mean()
                avg_cost = product_data['Cost'].mean()
            else:
                avg_price = 1000
                avg_cost = 700
            
            # Расчет выручки и маржи
            revenue = actual_sales * avg_price
            margin = actual_sales * (avg_price - avg_cost) - promo_cost
            
            # Обновляем остатки
            self.stocks[store][product] = current_stock - actual_sales
            
            # Награда = маржа
            reward = margin / 1000  # нормализуем
            
            self.total_revenue += revenue
            self.total_margin += margin
        
        self.current_step += 1
        self.actions_history.append({
            'step': self.current_step,
            'store': store,
            'product': product,
            'qty': qty,
            'promo': promo,
            'reward': reward
        })
        
        terminated = self.current_step >= self.horizon_days
        truncated = False
        
        return self._get_state(), reward, terminated, truncated, {}
    
    def render(self):
        """Визуализация состояния"""
        pass

class SimpleRLAgent:
    """Простой RL агент (Random baseline)"""
    
    def __init__(self, env):
        self.env = env
        self.q_table = {}
    
    def get_action(self, state):
        """Выбор действия (случайное)"""
        return self.env.action_space.sample()
    
    def train(self, episodes=100):
        """Обучение агента"""
        rewards_history = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
            
            rewards_history.append(total_reward)
            
            # Обновление прогресса
            progress_bar.progress((episode + 1) / episodes)
            status_text.text(f"Эпизод {episode + 1}/{episodes} | Награда: {total_reward:.2f}")
        
        progress_bar.empty()
        status_text.empty()
        
        return rewards_history

# Основное приложение
def main():
    # Боковая панель
    st.sidebar.header("⚙️ Настройки")
    
    # Загрузка файла
    uploaded_file = st.sidebar.file_uploader(
        "📁 Загрузите файл Excel с данными",
        type=['xlsx', 'xls'],
        help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
    )
    
    if uploaded_file is None:
        st.warning("⚠️ Пожалуйста, загрузите файл Excel с данными о продажах")
        st.info("""
        **Требуемая структура файла:**
        - Magazin - название магазина
        - Datasales - дата продажи
        - Art - артикул товара
        - Describe - описание
        - Model - модель
        - Segment - сегмент
        - Price - цена
        - Qty - количество
        - Sum - сумма
        """)
        st.stop()
    
    # Загрузка данных
    with st.spinner("Загрузка данных..."):
        df, store_features = load_and_prepare_data(uploaded_file)
    
    st.sidebar.success(f"✅ Загружено {len(df):,} записей")
    st.sidebar.info(f"📅 Период: {df['Datasales'].min().date()} - {df['Datasales'].max().date()}")
    
    # Табы
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Данные", 
        "🎯 RL Модель", 
        "📈 Результаты",
        "💡 Рекомендации"
    ])
    
    # TAB 1: Данные
    with tab1:
        st.header("Обзор данных")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Магазинов", df['Magazin'].nunique())
        with col2:
            st.metric("Товаров", df['Art'].nunique())
        with col3:
            st.metric("Общая выручка", f"{df['Sum'].sum():,.0f} ₴")
        with col4:
            st.metric("Средняя маржа", f"{df['Margin'].mean():.0f} ₴")
        
        st.subheader("Пример данных")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Визуализация
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Топ-10 магазинов по выручке")
            top_stores = df.groupby('Magazin')['Sum'].sum().nlargest(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_stores.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Выручка (₴)')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Распределение по сегментам")
            segment_sales = df.groupby('Segment')['Sum'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_ylabel('')
            st.pyplot(fig)
    
    # TAB 2: RL Модель
    with tab2:
        st.header("Обучение RL агента")
        
        col1, col2 = st.columns(2)
        
        with col1:
            episodes = st.slider("Количество эпизодов", 10, 500, 100)
            horizon_days = st.slider("Горизонт планирования (дней)", 7, 90, 30)
        
        with col2:
            st.info("""
            **Что делает агент:**
            - Распределяет товар между магазинами
            - Решает, когда запускать промо-акции
            - Максимизирует маржу за период
            """)
        
        if st.button("🚀 Запустить обучение", type="primary"):
            # Подготовка среды
            stores = df['Magazin'].unique()[:10]  # Берем 10 магазинов
            products = df['Art'].dropna().unique()
            
            env = RetailEnvironment(df, stores, products, horizon_days)
            agent = SimpleRLAgent(env)
            
            st.info("Обучение агента...")
            rewards = agent.train(episodes)
            
            # Сохраняем в session state
            st.session_state['rewards'] = rewards
            st.session_state['env'] = env
            st.session_state['agent'] = agent
            
            st.success("✅ Обучение завершено!")
            
            # График обучения
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(rewards, linewidth=2)
            ax.set_xlabel('Эпизод')
            ax.set_ylabel('Суммарная награда')
            ax.set_title('Кривая обучения RL агента')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # TAB 3: Результаты
    with tab3:
        st.header("Результаты и метрики")
        
        if 'rewards' in st.session_state:
            rewards = st.session_state['rewards']
            env = st.session_state['env']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Средняя награда (последние 10)",
                    f"{np.mean(rewards[-10:]):.2f}",
                    delta=f"{np.mean(rewards[-10:]) - np.mean(rewards[:10]):.2f}"
                )
            
            with col2:
                st.metric(
                    "Максимальная награда",
                    f"{max(rewards):.2f}"
                )
            
            with col3:
                st.metric(
                    "Улучшение",
                    f"{((np.mean(rewards[-10:]) / np.mean(rewards[:10]) - 1) * 100):.1f}%"
                )
            
            # Прогресс обучения
            st.subheader("Динамика обучения")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # График наград
            ax1.plot(rewards, alpha=0.6, linewidth=1, label='Награда за эпизод')
            
            # Скользящее среднее
            window = 10
            if len(rewards) > window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), moving_avg, 
                        linewidth=2, color='red', label=f'Скользящее среднее ({window})')
            
            ax1.set_xlabel('Эпизод')
            ax1.set_ylabel('Награда')
            ax1.set_title('Награды по эпизодам')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Гистограмма наград
            ax2.hist(rewards, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Награда')
            ax2.set_ylabel('Частота')
            ax2.set_title('Распределение наград')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning("⚠️ Сначала запустите обучение на вкладке 'RL Модель'")
    
    # TAB 4: Рекомендации
    with tab4:
        st.header("Рекомендации системы")
        
        if 'env' in st.session_state:
            env = st.session_state['env']
            
            st.subheader("📋 История действий (последние 20)")
            
            if len(env.actions_history) > 0:
                actions_df = pd.DataFrame(env.actions_history[-20:])
                actions_df['promo'] = actions_df['promo'].map({0: '❌ Нет', 1: '✅ Да'})
                st.dataframe(actions_df, use_container_width=True)
                
                # Статистика по промо
                st.subheader("📊 Анализ промо-акций")
                
                actions_full = pd.DataFrame(env.actions_history)
                promo_stats = actions_full.groupby('promo').agg({
                    'reward': ['mean', 'sum', 'count']
                }).round(2)
                
                st.dataframe(promo_stats)
                
                # Топ магазины
                st.subheader("🏆 Топ магазины по награде")
                store_stats = actions_full.groupby('store')['reward'].sum().nlargest(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                store_stats.plot(kind='barh', ax=ax, color='green')
                ax.set_xlabel('Суммарная награда')
                st.pyplot(fig)
                
            else:
                st.info("История действий будет доступна после обучения")
        else:
            st.warning("⚠️ Сначала запустите обучение на вкладке 'RL Модель'")
        
        # Общие рекомендации
        st.subheader("💡 Общие рекомендации")
        st.markdown("""
        **Следующие шаги для улучшения системы:**
        
        1. **Более сложный агент**: Использовать DQN, PPO или A3C вместо случайных действий
        2. **Больше признаков**: Добавить сезонность, конкурентов, погоду
        3. **Реальное A/B тестирование**: Проверить на небольшой группе магазинов
        4. **Непрерывное обучение**: Обновлять модель по мере поступления новых данных
        5. **Интеграция**: Подключить к системе управления складом
        """)

if __name__ == "__main__":
    main()