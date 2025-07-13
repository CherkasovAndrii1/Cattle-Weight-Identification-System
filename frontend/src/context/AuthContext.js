// src/context/AuthContext.js
import React, { createContext, useState, useContext, useEffect, useCallback, useMemo } from 'react';
// Убедитесь, что все эти функции существуют и экспортируются из вашего api.js
import { loginUserAPI, registerUserAPI, getCurrentUserAPI } from '../services/api.js';

// Создаем контекст
const AuthContext = createContext();

// Создаем провайдер контекста
export function AuthProvider({ children }) {
    // --- Состояния ---
    const [token, setToken] = useState(() => localStorage.getItem('accessToken'));
    const [user, setUser] = useState(null); // { id, email }
    const [isAuthenticated, setIsAuthenticated] = useState(false); // Начинаем с false до проверки
    const [isLoading, setIsLoading] = useState(true); // Начинаем с true, пока проверяем токен
    const [authError, setAuthError] = useState(null);

    // --- Эффект для проверки токена при загрузке приложения ---
    useEffect(() => {
        const verifyTokenAndFetchUser = async () => {
            const storedToken = localStorage.getItem('accessToken'); // Читаем токен здесь
            if (storedToken) {
                console.log("AuthProvider: Token found, verifying...");
                try {
                    // Пытаемся получить данные пользователя по токену
                    const userData = await getCurrentUserAPI(storedToken);
                    if (userData && userData.id) { // Проверяем, что данные получены
                        setUser(userData);
                        setToken(storedToken); // Устанавливаем токен в состояние
                        setIsAuthenticated(true);
                        console.log("AuthProvider: Token verified, user set:", userData);
                    } else {
                        // Если API вернуло что-то не то, но без ошибки
                        throw new Error("Invalid user data received");
                    }
                } catch (error) {
                    // Если токен невалиден (ошибка от getCurrentUserAPI, например 401/422)
                    console.error("AuthProvider: Token verification failed:", error);
                    localStorage.removeItem('accessToken'); // Удаляем невалидный токен
                    setToken(null);
                    setUser(null);
                    setIsAuthenticated(false);
                }
            } else {
                console.log("AuthProvider: No token found.");
                // Если токена нет, убедимся, что состояние чистое
                setToken(null);
                setUser(null);
                setIsAuthenticated(false);
            }
            // Завершаем начальную загрузку/проверку
            setIsLoading(false);
            console.log("AuthProvider: Initial loading finished.");
        };

        verifyTokenAndFetchUser();
        // Этот useEffect должен выполняться только один раз при монтировании
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Пустой массив зависимостей

    // --- Функции ---

    // Функция входа
    const login = useCallback(async (credentials) => {
        setIsLoading(true);
        setAuthError(null);
        try {
            const data = await loginUserAPI(credentials);
            if (data.access_token) {
                const newToken = data.access_token;
                localStorage.setItem('accessToken', newToken);
                setToken(newToken);
                setIsAuthenticated(true); // Сначала ставим true

                // Сразу получаем данные пользователя
                try {
                    console.log("AuthProvider: Fetching user data after login...");
                    const userData = await getCurrentUserAPI(newToken);
                    if (userData && userData.id) {
                         setUser(userData);
                         console.log("AuthProvider: User data fetched after login:", userData);
                    } else {
                        throw new Error("Invalid user data received after login");
                    }
                } catch (userError) {
                    console.error("AuthProvider: Failed to fetch user data after login:", userError);
                    // Не удалось получить данные пользователя, но вход был успешен.
                    // Выходим из системы, чтобы избежать несогласованного состояния.
                    logout(); // Вызываем logout, который очистит все
                    setAuthError('Login succeeded, but failed to fetch user data.');
                    setIsLoading(false);
                    return false;
                }

                setAuthError(null);
                setIsLoading(false);
                return true; // Успешный вход
            } else {
                throw new Error("Access token not received");
            }
        } catch (error) {
            console.error("AuthProvider: Login failed:", error);
            logout(); // Вызываем logout для очистки состояния
            setAuthError(error?.error || error?.message || 'Login failed');
            setIsLoading(false);
            return false; // Ошибка входа
        }
    }, []); // Добавим logout как зависимость? Нет, logout сам использует useCallback

    // Функция выхода
    const logout = useCallback(() => {
        console.log("AuthProvider: Logging out...");
        localStorage.removeItem('accessToken');
        setToken(null);
        setUser(null);
        setIsAuthenticated(false);
        setAuthError(null);
        // Можно добавить принудительную перезагрузку или перенаправление
        // window.location.href = '/login'; // Жесткая перезагрузка
    }, []);

     // Функция регистрации
     const register = useCallback(async (credentials) => {
        setIsLoading(true);
        setAuthError(null);
        try {
            const data = await registerUserAPI(credentials);
            console.log("AuthProvider: Registration successful:", data.message);
            setAuthError(null);
            setIsLoading(false);
            return true; // Успешная регистрация (не логиним автоматически)
        } catch (error) {
            console.error("AuthProvider: Registration failed:", error);
             setAuthError(error?.error || error?.message || 'Registration failed');
             setIsLoading(false);
             return false; // Ошибка регистрации
        }
    }, []);


    // --- Предоставляемое значение контекста ---
    // Передаем стабильные функции login, logout, register
    const value = useMemo(() => ({
        token,
        user,
        isAuthenticated,
        isLoading,
        authError,
        login,
        logout,
        register
    }), [token, user, isAuthenticated, isLoading, authError, login, logout, register]);

    // Не рендерим дочерние элементы, пока идет начальная проверка токена
    // чтобы избежать мигания UI (показ логина, а потом сразу главной страницы)
    // if (isLoading) {
    //     return <div>Loading Application...</div>; // Или ваш компонент спиннера
    // }

    return (
        <AuthContext.Provider value={value}>
            {/* Рендерим дочерние элементы только после завершения начальной загрузки */}
            {!isLoading ? children : <div>Loading Application...</div> /* Или спиннер */}
        </AuthContext.Provider>
    );
}

// --- Хук для удобного использования контекста ---
export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}