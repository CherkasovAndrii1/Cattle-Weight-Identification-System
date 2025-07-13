// src/App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// --- MUI ---
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container'; // Используем Container для основного контента
import Box from '@mui/material/Box'; // Используем Box для внешнего контейнера

// --- Компоненты ---
import Header from './components/Header';
import Footer from './components/Footer';
import NotificationSystem from './components/NotificationSystem';
// --- Страницы ---
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import SegmentationPage from './pages/SegmentationPage';
import KeypointsPage from './pages/KeypointsPage';
import WeightPage from './pages/WeightPage';
import NewsPage from './pages/NewsPage';
import HistoryPage from './pages/HistoryPage';
import NotFoundPage from './pages/NotFoundPage';
// --- Защищенный Маршрут ---
import ProtectedRoute from './components/ProtectedRoute';
// --- Хуки ---
import { useNotification } from './hooks/useNotification';

// import './styles/global.css'; // Можно удалить или оставить для специфичных глобальных стилей, не перекрываемых MUI
// import './styles/animations.css';

// --- 1. Создаем темную тему MUI ---
const darkTheme = createTheme({
  palette: {
    mode: 'dark', // Включаем темный режим
    // Можно настроить основные цвета, если стандартные не подходят
    // primary: {
    //   main: '#90caf9',
    // },
    // secondary: {
    //   main: '#f48fb1',
    // },
    // background: {
    //   default: '#121212', // Основной фон приложения
    //   paper: '#1e1e1e',   // Фон для "бумажных" элементов (карточки, формы и т.д.)
    // },
    // text: {
    //   primary: '#e0e0e0',
    //   secondary: '#b0b0b0',
    // }
  },
  // Можно добавить глобальные стили для компонентов, типографику и т.д.
  // components: {
  //   MuiButton: {
  //     styleOverrides: {
  //       root: {
  //         borderRadius: 8, // Пример: сделать все кнопки более скругленными
  //       },
  //     },
  //   },
  // },
});


function App() {
    const { notification, showNotification } = useNotification();

    // Удаляем инлайн стили, т.к. MUI будет управлять ими через ThemeProvider и CssBaseline
    // const appContainerStyle = { display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: '#121212', color: '#e0e0e0' };
    // const mainStyle = { flex: '1', width: '100%', maxWidth: '1200px', margin: '24px auto', padding: '0 16px' };

    return (
        // --- 2. Оборачиваем все в ThemeProvider ---
        <ThemeProvider theme={darkTheme}>
            {/* --- 3. Добавляем CssBaseline для сброса стилей и применения фона темы --- */}
            <CssBaseline />
            <BrowserRouter>
                {/* --- 4. Используем Box для flex-контейнера --- */}
                <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
                    <NotificationSystem notification={notification} />
                    <Header /> {/* Header остается общим (его тоже нужно будет адаптировать под MUI) */}

                    {/* --- 5. Используем Container для основного контента --- */}
                    <Container component="main" maxWidth="lg" sx={{ flexGrow: 1, py: 3 /* Вертикальные отступы */ }}>
                        <Routes>
                            {/* --- Публичные маршруты --- */}
                            <Route path="/login" element={<LoginPage />} />
                            <Route path="/register" element={<RegisterPage />} />

                            {/* --- Защищенные маршруты Обработки --- */}
                            <Route path="/segmentation" element={
                                <ProtectedRoute>
                                    <SegmentationPage showNotification={showNotification} />
                                </ProtectedRoute>
                            } />
                            <Route path="/keypoints" element={
                                <ProtectedRoute>
                                    <KeypointsPage showNotification={showNotification} />
                                </ProtectedRoute>
                            } />
                            <Route path="/weight" element={
                                <ProtectedRoute>
                                    <WeightPage showNotification={showNotification} />
                                </ProtectedRoute>
                            } />

                            {/* --- Другие Защищенные маршруты --- */}
                            <Route path="/history" element={
                                <ProtectedRoute>
                                    <HistoryPage />
                                </ProtectedRoute>
                            } />

                            {/* --- Другие маршруты --- */}
                            <Route path="/news" element={<NewsPage />} />

                            {/* --- Редирект с главной --- */}
                            <Route path="/" element={<Navigate to="/segmentation" replace />} />

                            {/* --- Страница не найдена --- */}
                            <Route path="*" element={<NotFoundPage />} />
                        </Routes>
                    </Container>

                    <Footer /> {/* Footer остается общим (его тоже нужно будет адаптировать под MUI) */}
                </Box>
            </BrowserRouter>
        </ThemeProvider>
    );
}

export default App;