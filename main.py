import requests
import pandas as pd
from datetime import datetime, timedelta
import random

print("🚀 Запуск процесса ETL и обработки данных...")

# ==========================================
# 1. ПОЛУЧЕНИЕ ДАННЫХ ИЗ API (DummyJSON)
# ==========================================
BASE_URL = "https://dummyjson.com"

# Запрашиваем товары, корзины и пользователей
res_products = requests.get(f"{BASE_URL}/products?limit=0").json()
res_carts = requests.get(f"{BASE_URL}/carts?limit=0").json()
res_users = requests.get(f"{BASE_URL}/users?limit=0").json()

df_products_raw = pd.DataFrame(res_products['products'])
df_carts_raw = pd.DataFrame(res_carts['carts'])
df_users_raw = pd.DataFrame(res_users['users'])

print(f"✓ Загружено товаров: {len(df_products_raw)}")
print(f"✓ Загружено корзин: {len(df_carts_raw)}")
print(f"✓ Загружено пользователей: {len(df_users_raw)}")

# ==========================================
# 2. ОБРАБОТКА И ПОДГОТОВКА СТРУКТУРЫ ДАННЫХ
# ==========================================

# --- 2.1 Таблица продуктов (dim_products) ---
df_products = df_products_raw[
    ['id', 'title', 'category', 'price', 'discountPercentage', 'rating', 'stock', 'brand']].copy()
df_products.rename(columns={'id': 'product_id'}, inplace=True)
df_products['brand'] = df_products['brand'].fillna('Generic')

# --- 2.2 Таблица пользователей (dim_users) ---
df_users = pd.DataFrame()
df_users['user_id'] = df_users_raw['id']
df_users['full_name'] = df_users_raw['firstName'] + " " + df_users_raw['lastName']
df_users['age'] = df_users_raw['age']
df_users['gender'] = df_users_raw['gender']
df_users['city'] = df_users_raw['address'].apply(lambda x: x.get('city') if isinstance(x, dict) else 'Unknown')
df_users['country'] = df_users_raw['address'].apply(lambda x: x.get('country', 'USA') if isinstance(x, dict) else 'USA')

# --- 2.3 Транзакции продаж (fact_sales) ---
sales_list = []
# Создаем реалистичные даты заказов за последние 365 дней
now = datetime.now()

for cart in res_carts['carts']:
    cart_id = cart['id']
    user_id = cart['userId']

    # Эмуляция даты заказа для тайм-серий
    random_days_ago = random.randint(1, 365)
    order_date = (now - timedelta(days=random_days_ago)).strftime('%Y-%m-%d')

    for item in cart['products']:
        sales_list.append({
            'cart_id': cart_id,
            'user_id': user_id,
            'product_id': item['id'],
            'quantity': item['quantity'],
            'unit_price': item['price'],
            'total_price': item['total'],
            'discounted_price': item['discountedTotal'],
            'order_date': order_date
        })

df_sales = pd.DataFrame(sales_list)
df_sales['order_date'] = pd.to_datetime(df_sales['order_date'])

# ==========================================
# 3. RFM-АНАЛИЗ КЛИЕНТОВ (Расчет метрик)
# ==========================================
# Snapshot date — текущий день
snapshot_date = df_sales['order_date'].max() + timedelta(days=1)

# Группировка по пользователям
rfm = df_sales.groupby('user_id').agg({
    'order_date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'cart_id': 'nunique',  # Frequency
    'discounted_price': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['user_id', 'recency_days', 'frequency', 'monetary']

# Присвоение баллов от 1 до 4 (Квантили)
rfm['R_score'] = pd.qcut(rfm['recency_days'], q=4, labels=[4, 3, 2, 1])
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
rfm['M_score'] = pd.qcut(rfm['monetary'], q=4, labels=[1, 2, 3, 4])


# Сегментация пользователей
def rfm_segment(row):
    r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
    if r >= 3 and f >= 3 and m >= 3:
        return 'VIP / Champions'
    elif r >= 3 and f >= 2:
        return 'Loyal Customers'
    elif r <= 2 and f >= 3:
        return 'At Risk (Unretained)'
    elif r <= 2 and f <= 2:
        return 'Lost Customers'
    else:
        return 'Promising / New'


rfm['rfm_segment'] = rfm.apply(rfm_segment, axis=1)

# Объединяем RFM с таблицей пользователей
df_users = df_users.merge(rfm[['user_id', 'recency_days', 'frequency', 'monetary', 'rfm_segment']], on='user_id',
                          how='left')

# ==========================================
# 4. ИИ-КАТЕГОРИЗАЦИЯ ОТЗЫВОВ (AI Analysis)
# ==========================================
print("🧠 Запуск ИИ-классификации отзывов о товарах...")

reviews_list = []
for product in res_products['products']:
    p_id = product['id']
    reviews = product.get('reviews', [])
    for r in reviews:
        reviews_list.append({
            'product_id': p_id,
            'reviewer_name': r.get('reviewerName'),
            'rating': r.get('rating'),
            'comment': r.get('comment'),
            'review_date': r.get('date')
        })

df_reviews = pd.DataFrame(reviews_list)


# Алгоритм ИИ-анализа тональности и причин недовольства
def ai_classifier(comment, rating):
    comment_lower = str(comment).lower()

    if rating >= 4:
        sentiment = 'Positive'
        issue_category = 'None (Satisfied)'
    else:
        sentiment = 'Negative'
        if any(w in comment_lower for w in ['shipping', 'delivery', 'late', 'arrive', 'package', 'delay']):
            issue_category = 'Logistics & Shipping'
        elif any(w in comment_lower for w in ['quality', 'broken', 'cheap', 'bad', 'disappointed', 'damaged']):
            issue_category = 'Product Quality'
        elif any(w in comment_lower for w in ['price', 'expensive', 'worth', 'cost']):
            issue_category = 'Pricing & Value'
        else:
            issue_category = 'Customer Service & Misc'

    return pd.Series([sentiment, issue_category])


# Применяем ИИ-разметку
df_reviews[['ai_sentiment', 'ai_issue_category']] = df_reviews.apply(
    lambda x: ai_classifier(x['comment'], x['rating']), axis=1
)

# ==========================================
# 5. ЭКСПОРТ В CSV ДЛЯ POWER BI
# ==========================================
df_sales.to_csv('fact_sales.csv', index=False, encoding='utf-8-sig')
df_products.to_csv('dim_products.csv', index=False, encoding='utf-8-sig')
df_users.to_csv('dim_users.csv', index=False, encoding='utf-8-sig')
df_reviews.to_csv('fact_reviews_ai.csv', index=False, encoding='utf-8-sig')

print("\n🎉 ВСЕ ДАННЫЕ УСПЕШНО ОБРАБОТАНЫ И СОХРАНЕНЫ!")
print("Файлы для Power BI созданы в текущей папке:")
print(" 1. fact_sales.csv (Факты продаж)")
print(" 2. dim_products.csv (Справочник товаров)")
print(" 3. dim_users.csv (Справочник клиентов с RFM-сегментацией)")
print(" 4. fact_reviews_ai.csv (ИИ-аналитика отзывов)")