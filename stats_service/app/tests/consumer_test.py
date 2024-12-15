import pytest
from unittest.mock import AsyncMock
from consumer.consumer import consume_clicks

@pytest.mark.asyncio
async def test_consume_clicks_with_error():
    # Создаем мок для dao
    mock_dao = AsyncMock()
    mock_dao.process_message = AsyncMock()
    mock_dao.insert_batch = AsyncMock()

    # Создаем мок для consume
    mock_consume = AsyncMock()
    mock_consume.start = AsyncMock()
    mock_consume.stop = AsyncMock()

    # Настроим список как итерабельный объект
    mock_consume.__aiter__.return_value = iter([{"value": {"user_id": 1, "movie_id": 101}}])

    # Запускаем тестируемую функцию
    await consume_clicks(mock_consume, mock_dao)

    # Проверяем вызовы consume
    mock_consume.start.assert_called_once()
    mock_consume.stop.assert_called_once()

    # Допустим, что ошибка должна возникнуть, если process_message не был вызван
    with pytest.raises(AssertionError):
        mock_dao.process_message.assert_called_once_with({"value": {"user_id": 2, "movie_id": 102}})  # Некорректное значение
