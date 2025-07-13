// src/components/ProtectedRoute.jsx
import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext'; // ПРАВИЛЬНО

function ProtectedRoute({ children }) {
    const { isAuthenticated, isLoading } = useAuth();
    const location = useLocation(); // Запоминаем, куда пользователь хотел попасть

    // Если все еще идет проверка токена в AuthContext, показываем заглушку
    if (isLoading) {
        // Можно вернуть null или полноценный спиннер/заглушку
        return <div>Проверка аутентификации...</div>;
    }

    // Если проверка завершена и пользователь НЕ аутентифицирован
    if (!isAuthenticated) {
        // Перенаправляем на страницу входа, сохраняя путь, куда пользователь шел
        console.log('ProtectedRoute: User not authenticated, redirecting to /login');
        // `replace` заменяет текущую запись в истории, чтобы кнопка "назад" работала адекватно
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    // Если проверка завершена и пользователь аутентифицирован, рендерим дочерний компонент
    return children;
}

export default ProtectedRoute; // <-- Убедитесь, что экспорт по умолчанию есть