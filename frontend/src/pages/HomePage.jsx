// src/pages/HomePage.jsx
import React, { useState, useCallback, useEffect } from 'react'; // <--- Добавлен useEffect
// --- Хуки React Router ---
import { useLocation, useNavigate } from 'react-router-dom'; // <--- Добавлены useLocation и useNavigate

// --- Компоненты UI ---
import TabMenu from '../components/TabMenu/TabMenu';         // <-- Убедитесь, что путь верный
import ImageUploader from '../components/ImageUploader/ImageUploader'; // <-- Убедитесь, что путь верный
import ResultViewer from '../components/ResultViewer/ResultViewer';   // <-- Убедитесь, что путь верный
// --- API и Константы ---
import { processImageAPI } from '../services/api';
import { TAB_KEYS, TAB_NAMES } from '../constants/tabNames';
// --- (Опционально) Контекст/Хуки ---
// import { useAuth } from '../context/AuthContext';
// import { useNotification } from '../hooks/useNotification'; // Если не передается через пропс

function HomePage({ showNotification }) {
    // --- Состояния для основного интерфейса ---
    const [activeTab, setActiveTab] = useState(TAB_KEYS.SEGMENTATION);
    const [selectedFile, setSelectedFile] = useState(null);
    // --- ВАЖНО: Инициализируем processedImage из location.state ЕСЛИ ОНО ЕСТЬ ---
    // Это позволит сразу показать историю, если мы перешли с HistoryPage
    const location = useLocation();
    const initialHistoryItem = location.state?.historyItemToLoad || null;
    const [processedImage, setProcessedImage] = useState(initialHistoryItem);
    // ---------------------------------------------------------------------
    const [isProcessing, setIsProcessing] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    // const { user } = useAuth();
    const navigate = useNavigate(); // <--- Получаем функцию навигации ЗДЕСЬ

    // --- Эффект для очистки location.state после загрузки истории ---
    useEffect(() => {
        // Если мы загрузили что-то из state, очищаем его, чтобы при
        // обновлении страницы или обычном заходе оно не загружалось снова
        if (location.state?.historyItemToLoad) {
            console.log("Loaded item from history state:", location.state.historyItemToLoad);
            // Очищаем state в истории браузера, не меняя URL
            navigate(location.pathname, { replace: true, state: {} });
        }
        // Зависимость только от location.state, чтобы сработал один раз при изменении state
    }, [location.state, navigate]);


    // --- Обработчики (без изменений) ---
    const handleProcessImage = useCallback(async (file, processType) => {
        // ... (код как и раньше) ...
         if (!showNotification) console.error("showNotification function is not provided to HomePage");
         if (!file) return;
         setIsProcessing(true);
         setProcessedImage(null);
         const reader = new FileReader();
         reader.readAsDataURL(file);
         reader.onload = async (e) => {
             const originalImageBase64 = e.target.result;
             try {
                 const apiResult = await processImageAPI(file, processType);
                 const result = {
                      id: Date.now(), type: processType, originalImage: originalImageBase64,
                      timestamp: new Date().toLocaleString(), processingTime: apiResult.processing_time || 'N/A',
                      filename: file.name, processedImage: apiResult.image_url || null,
                      stats: apiResult.stats || null, keypoints: apiResult.keypoints || null,
                      estimatedWeightKg: apiResult.estimated_weight_kg != null ? apiResult.estimated_weight_kg : null,
                      originalWidth: apiResult.original_width || null, originalHeight: apiResult.original_height || null,
                 };
                 setProcessedImage(result);
                 if (showNotification) showNotification(`Обработка "${TAB_NAMES[processType]}" завершена!`, 'success');
             } catch (error) {
                  console.error("Error processing image in HomePage:", error);
                  const errorMessage = error?.error || error?.message || 'Произошла ошибка при обработке';
                  if (showNotification) showNotification(`Ошибка: ${errorMessage}`, 'error');
                  if (error?.response?.status === 401 || error?.response?.status === 403) { /*...*/ }
             } finally { setIsProcessing(false); }
         };
         reader.onerror = (error) => {
             console.error("Ошибка чтения файла:", error);
             if (showNotification) showNotification('Не удалось прочитать файл', 'error');
             setIsProcessing(false);
         };
     }, [showNotification]);

     const handleSubmit = useCallback((e) => {
         e.preventDefault();
         if (selectedFile && !isProcessing) {
             handleProcessImage(selectedFile, activeTab);
         } else if (!selectedFile) {
              if (showNotification) showNotification("Пожалуйста, сначала выберите файл.", 'warning');
         }
     }, [selectedFile, activeTab, handleProcessImage, isProcessing, showNotification]);

     const handleFileChange = useCallback((e) => {
         if (e.target.files && e.target.files[0]) {
             const file = e.target.files[0];
             if (file.type.startsWith('image/')) {
                 setSelectedFile(file);
                 setProcessedImage(null); 
                 if (showNotification) showNotification(`File "${file.name}" chosen`, 'info');
             } else {
                  if (showNotification) showNotification('Please, choose an image file', 'warning');
             }
         }
     }, [showNotification]);

    const handleDragEnter = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }, []);
    const handleDragLeave = useCallback((e) => { e.preventDefault(); e.stopPropagation(); if (e.currentTarget.contains(e.relatedTarget)) return; setIsDragging(false); }, []);
    const handleDragOver = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }, []);
    const handleDrop = useCallback((e) => {
        e.preventDefault(); e.stopPropagation(); setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
             const file = e.dataTransfer.files[0];
             if (file.type.startsWith('image/')) {
                 setSelectedFile(file);
                 setProcessedImage(null); // Сбрасываем результат при дропе нового файла
                 if (showNotification) showNotification(`Файл "${file.name}" выбран`, 'info');
             } else {
                  if (showNotification) showNotification('Пожалуйста, выберите файл изображения.', 'warning');
             }
         }
    }, [showNotification]);

    const handleUploadAreaClick = useCallback(() => {
        const fileInput = document.getElementById('file-upload');
        if (fileInput) { fileInput.click(); }
    }, []);

    const handleSetActiveProcessingTab = useCallback((tabKey) => {
        setActiveTab(tabKey);
        // Не сбрасываем файл при смене вкладки, только результат
        setProcessedImage(null);
    }, []);

    // --- Стили ---
    const mainProcessingStyle = { backgroundColor: '#1e1e1e', borderRadius: '16px', boxShadow: '0 8px 20px rgba(0,0,0,0.2)', padding: '24px', border: '1px solid #333', animation: 'fadeIn 0.5s ease-out' };
    const tabMenuStyle = { marginBottom: '20px' };

    // --- Рендеринг ---
    return (
        <div>
            <div style={tabMenuStyle}>
                 <TabMenu
                    activeProcessingTab={activeTab}
                    setActiveProcessingTab={handleSetActiveProcessingTab}
                />
            </div>

            <div style={mainProcessingStyle}>
                 <ImageUploader
                    activeTab={activeTab}
                    selectedFile={selectedFile}
                    isProcessing={isProcessing}
                    isDragging={isDragging}
                    handleSubmit={handleSubmit}
                    handleFileChange={handleFileChange}
                    handleDragEnter={handleDragEnter}
                    handleDragLeave={handleDragLeave}
                    handleDragOver={handleDragOver}
                    handleDrop={handleDrop}
                    handleUploadAreaClick={handleUploadAreaClick}
                />
                {/* Отображаем результат (может быть из истории или новой обработки) */}
                {processedImage && <ResultViewer processedImage={processedImage} />}
                {isProcessing && <div style={{ textAlign: 'center', margin: '20px 0', color: '#bb86fc' }}>Processing...</div>}
            </div>
        </div>
    );
}

export default HomePage;