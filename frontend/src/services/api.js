import axios from 'axios';
// Убедитесь, что эти файлы существуют и правильно экспортируют константы
import { API_ENDPOINTS, API_BASE_URL } from '../constants/apiEndpoints'; // Предполагаем наличие этих констант

const apiClient = axios.create({
    baseURL: `${API_BASE_URL}/api/v1`, 
    timeout: 60000, 
});

// --- Перехватчик Запросов (Без изменений, он корректен) ---
apiClient.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('accessToken');
        const noAuthUrls = ['/auth/login', '/auth/register', '/health']; 

        const requestPath = config.url; 

        if (token && requestPath && !noAuthUrls.some(url => requestPath.startsWith(url))) {
            config.headers['Authorization'] = `Bearer ${token}`;
            console.log(`Interceptor: Added token to request for ${config.baseURL}${requestPath}`);
        } else {
             console.log(`Interceptor: No token added for ${config.baseURL}${requestPath}`);
        }
        return config;
    },
    (error) => {
        console.error("Interceptor request error:", error);
        return Promise.reject(error);
    }
);

// --- Перехватчик Ответов (Без изменений, он корректен) ---
apiClient.interceptors.response.use(
    (response) => response, // Просто возвращаем успешный ответ
    (error) => {
        if (error.response && error.response.status === 401) {
            console.warn("Interceptor response: Received 401 Unauthorized. Clearing token.");
            localStorage.removeItem('accessToken');
            // Можно добавить вызов функции logout из контекста, если передать его сюда,
            // или просто перезагрузить страницу, чтобы пользователь попал на логин
            // window.location.href = '/login';
        }
        // Важно перебросить ошибку дальше
        return Promise.reject(error);
    }
);


/**
 * Отправляет изображение на бэкенд для обработки.
 * @param {File} file - Файл изображения.
 * * @param {string} processType - Тип обработки ('segmentation', 'keypoints', 'weight').
 * @returns {Promise<object>} - Promise с объектом результата от бэкенда.
 * @throws {Error} - Выбрасывает ошибку при неудачном запросе или неверном ответе.
 */
export const processImageAPI = async (file, processType) => { // <-- Убираем token из аргументов
    const apiUrl = API_ENDPOINTS[processType]; // Получаем относительный путь
    if (!apiUrl) {
        throw new Error(`Неверный тип обработки: ${processType}`);
    }

    const formData = new FormData();
    formData.append('file', file);

    console.log(`Отправка запроса через apiClient на ${apiUrl} с файлом: ${file.name}`);

    try {
        // <<< ИЗМЕНЕНО: Используем apiClient вместо глобального axios >>>
        const response = await apiClient.post(apiUrl, formData, {
            // Заголовки 'Content-Type': 'multipart/form-data' и 'Authorization'
            // будут добавлены axios и interceptor'ом соответственно.
            // Можно увеличить таймаут для конкретно этого запроса, если нужно
            // timeout: 120000 // 2 минуты
        });

        console.log(`Ответ от бэкенда (${processType}):`, response.data);

        // Проверяем успешность ответа от нашего бэкенда (поле success)
        if (response.data && response.data.success) {
            return response.data; // Возвращаем все данные
        } else {
            // Если бэкенд вернул success: false или не вернул success
            throw new Error(response.data?.error || response.data?.message || 'Некорректный или неуспешный ответ от сервера');
        }
    } catch (error) {
        console.error(`Ошибка API при обработке (${processType}):`, error);
        // Перехватчик ответа уже может обработать 401, но другие ошибки нужно обработать здесь
        if (error.response) {
            // Ошибка от бэкенда (статус не 2xx)
            throw new Error(`Ошибка сервера: ${error.response.data?.error || error.response.data?.message || error.response.status}`);
        } else if (error.request) {
            // Нет ответа
            throw new Error('Не удалось связаться с сервером.');
        } else {
            // Ошибка настройки запроса axios
            throw new Error(`Ошибка запроса: ${error.message}`);
        }
    }
};

// <<< НОВАЯ ФУНКЦИЯ >>>
/**
 * Запрашивает историю обработок для текущего пользователя.
 * @returns {Promise<Array>} - Promise со списком записей истории.
 * @throws {Error} - Выбрасывает ошибку при неудаче.
 */
export const getHistoryAPI = async () => {
    const apiUrl = '/history'; // Относительный путь к эндпоинту истории
    console.log(`Запрос истории через apiClient на ${apiUrl}`);
    try {
        // Используем apiClient, interceptor добавит токен
        const response = await apiClient.get(apiUrl);
        console.log('История получена:', response.data);
        // Бэкенд возвращает список напрямую в случае успеха
        return response.data;
    } catch (error) {
        console.error('Ошибка API при получении истории:', error);
        if (error.response) {
            throw new Error(`Ошибка сервера: ${error.response.data?.error || error.response.status}`);
        } else if (error.request) {
            throw new Error('Не удалось связаться с сервером.');
        } else {
            throw new Error(`Ошибка запроса: ${error.message}`);
        }
    }
};


// --- Функции getCurrentUserAPI, registerUserAPI, loginUserAPI остаются как были ---
// Они уже используют apiClient и interceptor'ы
export const getCurrentUserAPI = async () => {
    // Токен будет добавлен автоматически interceptor'ом
    console.log("Attempting to fetch current user data via apiClient...");
    try {
        const response = await apiClient.get('/users/me'); // Interceptor добавит 'Authorization: Bearer ...'
        console.log("User data received:", response.data);
        return response.data; // Ожидаем { id, email }
    } catch (error) {
        console.error("Error fetching current user:", error.response?.data || error.message);
        // Ошибка (401/422) будет обработана перехватчиком ответа и/или в AuthContext
        throw error.response?.data || new Error('Failed to fetch user data');
    }
};

export const registerUserAPI = async (credentials) => {
    try {
        // Используем apiClient, токен НЕ будет добавлен (т.к. URL /auth/register в списке исключений)
        const response = await apiClient.post('/auth/register', credentials);
        return response.data;
    } catch (error) {
        console.error("Error during user registration:", error.response?.data || error.message);
        throw error.response?.data || new Error('Registration failed');
    }
};

export const loginUserAPI = async (credentials) => {
    try {
         // Используем apiClient, токен НЕ будет добавлен (т.к. URL /auth/login в списке исключений)
        const response = await apiClient.post('/auth/login', credentials);
        if (!response.data || !response.data.access_token) {
            throw new Error("Invalid response format from login API");
        }
        return response.data;
    } catch (error) {
        console.error("Error during user login:", error.response?.data || error.message);
        throw error.response?.data || new Error('Login failed');
    }
};