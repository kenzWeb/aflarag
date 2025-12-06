from qdrant_client import QdrantClient, models

# НАСТРОЙКИ
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents1" # Убедись, что имя верное!
TEST_ID = 1157  # ID из твоего примера (первый в списке)

client = QdrantClient(url=QDRANT_URL)

def test_search(search_val):
    print(f"\n🔎 Пробуем найти doc_id = {search_val} (Тип: {type(search_val)})")
    
    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", 
                        match=models.MatchValue(value=search_val)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        if points:
            print(f"✅ УСПЕХ! Найдено чанков: {len(points)}")
            print(f"Пример текста: {points[0].payload.get('text', '')[:50]}...")
            return True
        else:
            print("❌ Ничего не найдено.")
            return False
            
    except Exception as e:
        print(f"Ошибка запроса: {e}")
        return False

# 1. Пробуем найти просто любой документ, чтобы посмотреть поля
print("--- ПРОВЕРКА СТРУКТУРЫ БАЗЫ ---")
res = client.scroll(collection_name=COLLECTION_NAME, limit=1, with_payload=True)
if res[0]:
    payload = res[0][0].payload
    print(f"Случайный документ в базе имеет поля: {list(payload.keys())}")
    print(f"Пример doc_id внутри: {payload.get('doc_id')} (Тип: {type(payload.get('doc_id'))})")
else:
    print("⚠️ База пуста или коллекция не найдена!")

# 2. Пробуем искать конкретный ID как число
test_search(TEST_ID)

# 3. Пробуем искать конкретный ID как строку
test_search(str(TEST_ID))