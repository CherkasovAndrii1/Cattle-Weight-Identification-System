// src/pages/HistoryPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
// <<< ИЗМЕНЕНО: Добавляем Button из MUI >>>
import { Box, Typography, CircularProgress, Alert, Button } from '@mui/material';
import HistoryPanel from '../components/HistoryPanel/HistoryPanel';
import { getHistoryAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
// <<< ИЗМЕНЕНО: Импорт иконки (опционально) >>>
import DownloadIcon from '@mui/icons-material/Download'; // Пример иконки

function HistoryPage() {
    const [historyData, setHistoryData] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const { token, isAuthenticated } = useAuth();
    const navigate = useNavigate();

    // Функция для загрузки истории (без изменений)
    const fetchHistory = useCallback(async () => {
        if (!isAuthenticated || !token) {
             console.log("HistoryPage: User not authenticated, skipping fetch.");
             setHistoryData([]);
             setError("You need to log in firstly");
             setIsLoading(false);
             return;
         }
        setIsLoading(true);
        setError(null);
        console.log("HistoryPage: Fetching history...");
        try {
            const data = await getHistoryAPI();
            setHistoryData(data || []);
            console.log("HistoryPage: History data loaded:", data);
        } catch (err) {
            console.error("HistoryPage: Failed to fetch history:", err);
            setError(err.message || 'Не удалось загрузить историю');
            setHistoryData([]);
        } finally {
            setIsLoading(false);
        }
    }, [token, isAuthenticated]);

    useEffect(() => {
        fetchHistory();
    }, [fetchHistory]);

    // Обработчик клика (без изменений)
    const loadHistoryItem = useCallback((item) => {
        console.log("Loading history item:", item);
        navigate('/', { state: { historyItemToLoad: item } });
    }, [navigate]);

    // <<< НОВАЯ ФУНКЦИЯ: Экспорт в JSON >>>
    const handleExportJson = useCallback(() => {
        if (!historyData || historyData.length === 0) {
            console.warn("No history data to export.");
            // Можно показать уведомление пользователю
            return;
        }

        try {
            const jsonString = JSON.stringify(historyData, null, 2); // null, 2 для форматирования с отступами

            const blob = new Blob([jsonString], { type: "application/json" });

            const url = URL.createObjectURL(blob);

            const link = document.createElement('a');
            link.href = url;
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            link.download = `image_processing_history_${timestamp}.json`;

            document.body.appendChild(link); 
            link.click();


            document.body.removeChild(link);
            URL.revokeObjectURL(url);

            console.log("History exported successfully.");

        } catch (err) {
            console.error("Failed to export history to JSON:", err);
        }

    }, [historyData]); 


    // --- Отображение ---
    if (isLoading) {
        return (
             <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <CircularProgress />
             </Box>
        );
    }

    // Ошибку показываем, если она есть, независимо от наличия данных
    if (error) {
        return <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>;
    }

    return (
        <Box sx={{ animation: 'fadeIn 0.5s ease-out' }}>
             {/* <<< ИЗМЕНЕНО: Добавлен Box для заголовка и кнопки >>> */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 0 }}>
                    Request history
                </Typography>
                 {/* <<< НОВАЯ КНОПКА ЭКСПОРТА >>> */}
                 <Button
                     variant="outlined"
                     startIcon={<DownloadIcon />} // Опциональная иконка
                     onClick={handleExportJson}
                     disabled={historyData.length === 0 || isLoading} // Деактивируем, если нет данных или идет загрузка
                 >
                     Export in JSON
                 </Button>
            </Box>

            {/* Отображаем панель или сообщение */}
            {historyData.length === 0 && !isLoading ? (
                 <Typography sx={{ mt: 3, textAlign: 'center' }}>
                    Your request history is empty
                 </Typography>
            ) : (
                 <HistoryPanel
                    history={historyData}
                    loadHistoryItem={loadHistoryItem}
                 />
                 // Если вы не используете HistoryPanel, а рендерите Grid/Card здесь,
                 // то этот код остается без изменений.
            )}
        </Box>
    );
}

export default HistoryPage;