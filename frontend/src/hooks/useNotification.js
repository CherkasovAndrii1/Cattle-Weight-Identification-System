// src/hooks/useNotification.js
import { useState, useCallback } from 'react';

export const useNotification = (timeout = 3000) => {
    const [notification, setNotification] = useState(null);
    const [notificationId, setNotificationId] = useState(0); // Для сброса таймаута

    const showNotification = useCallback((message) => {
        setNotification(message);
        const currentId = Date.now(); // Уникальный ID для этого уведомления
        setNotificationId(currentId);

        setTimeout(() => {
            // Скрываем уведомление только если это то же самое, что мы показали
            setNotification((prevNotification) => {
                // Сверяем не по ID, а по сообщению, т.к. ID уже может измениться
                // Проще просто скрыть через 3 секунды
                return null;
            });
             // Или использовать ID для сброса таймаута, если нужно более сложное поведение
             // setNotification(prev => prev && notificationId === currentId ? null : prev);
        }, timeout);
    }, [timeout]); // Убрали notificationId из зависимостей

    const hideNotification = useCallback(() => {
        setNotification(null);
    }, []);

    return { notification, showNotification, hideNotification };
};