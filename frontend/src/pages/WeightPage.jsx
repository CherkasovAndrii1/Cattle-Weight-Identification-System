// src/pages/WeightPage.jsx
import React, { useState, useCallback } from 'react';

// --- Компоненты, API, Константы, Хуки ---

import ImageUploader from '../components/ImageUploader/ImageUploader';
import ResultViewer from '../components/ResultViewer/ResultViewer';
import { processImageAPI } from '../services/api';
import { TAB_KEYS, TAB_NAMES } from '../constants/tabNames';
// import { useNotification } from '../hooks/useNotification';

function WeightPage({ showNotification }) {
    // const { showNotification } = useNotification();

    // --- Состояния ---
    const [selectedFile, setSelectedFile] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isDragging, setIsDragging] = useState(false);

    // --- Фиксированный тип обработки ---
    const pageProcessType = TAB_KEYS.WEIGHT; // <--- ОТЛИЧИЕ

    // --- Обработчики (Код обработчиков точно такой же, как в SegmentationPage,
    // за исключением того, что handleProcessImage использует pageProcessType = WEIGHT) ---

    const handleProcessImage = useCallback(async (file) => {
        if (!showNotification) console.error("showNotification function is not provided");
        if (!file) return;
        setIsProcessing(true); setProcessedImage(null);
        const reader = new FileReader(); reader.readAsDataURL(file);
        reader.onload = async (e) => {
             const originalImageBase64 = e.target.result;
             try {
                 const apiResult = await processImageAPI(file, pageProcessType); // Используется WEIGHT
                 const result = { /* ... формирование result (stats и keypoints будут null, а weight - получен) ... */
                      id: Date.now(), type: pageProcessType, originalImage: originalImageBase64,
                      timestamp: new Date().toLocaleString(), processingTime: apiResult.processing_time || 'N/A',
                      filename: file.name,
                      // Бэкенд для /weight может вернуть и URL картинки с весом (если он ее рисует)
                      processedImage: apiResult.image_url || apiResult.keypoints_image_url || null,
                      stats: null, keypoints: null, // Или apiResult.keypoints, если бэкенд их возвращает
                      estimatedWeightKg: apiResult.estimated_weight_kg != null ? apiResult.estimated_weight_kg : null, // Получаем вес
                      originalWidth: apiResult.original_width || null, originalHeight: apiResult.original_height || null,
                 };
                 setProcessedImage(result);
                 if (showNotification) showNotification(`Processing "${TAB_NAMES[pageProcessType]}" done!`, 'success');
             } catch (error) { /* ... обработка ошибок ... */
                 console.error(`Error processing ${pageProcessType}:`, error);
                  const errorMessage = error?.error || error?.message || 'Error during processing';
                  if (showNotification) showNotification(`Error: ${errorMessage}`, 'error');
             } finally { setIsProcessing(false); }
         };
         reader.onerror = (error) => { /* ... */ };
    }, [showNotification, pageProcessType]);

const handleSubmit = useCallback((e) => {
        e.preventDefault();
        if (selectedFile && !isProcessing) {
            handleProcessImage(selectedFile);
        } else if (!selectedFile) {
            if (showNotification) showNotification("Please select a file first.", 'warning');
        }
    }, [selectedFile, handleProcessImage, isProcessing, showNotification]);

    const handleFileChange = useCallback((e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            if (file.type.startsWith('image/')) {
                setSelectedFile(file);
                setProcessedImage(null);
                if (showNotification) showNotification(`File "${file.name}" chosen`, 'info');
            } else {
                // --- Полная обработка неверного типа файла ---
                 if (showNotification) showNotification('Please select an image file.', 'warning');
                 setSelectedFile(null); // Сбрасываем выбор
                 setProcessedImage(null);
                // ------------------------------------------
            }
        }
         // Сбрасываем значение инпута, чтобы можно было выбрать тот же файл снова
         if (e.target) {
           e.target.value = null;
         }
    }, [showNotification]);

    // Обработчики Drag & Drop (без изменений, они были полными)
    const handleDragEnter = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }, []);
    const handleDragLeave = useCallback((e) => { e.preventDefault(); e.stopPropagation(); if (e.currentTarget.contains(e.relatedTarget)) return; setIsDragging(false); }, []);
    const handleDragOver = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }, []);

    // --- Полный обработчик handleDrop ---
    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false); // Убираем стиль перетаскивания

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                setSelectedFile(file);
                setProcessedImage(null); // Сбрасываем предыдущий результат
                if (showNotification) showNotification(`File "${file.name}" successfully dragged in`, 'info');
            } else {
                if (showNotification) showNotification('You can only drag and drop image files.', 'warning');
                 setSelectedFile(null); // Сбрасываем выбор
                 setProcessedImage(null);
            }
        }
    }, [showNotification]);
    // --------------------------------

    // --- Полный обработчик handleUploadAreaClick ---
    const handleUploadAreaClick = useCallback(() => {
        // Находим скрытый инпут по ID
        const fileInput = document.getElementById('file-upload');
        // Если найден, имитируем клик по нему
        if (fileInput) {
            fileInput.click();
        } else {
            console.error('File input element with id="file-upload" was not found!');
            // Можно показать уведомление пользователю, если что-то пошло не так
            if(showNotification) showNotification('Failed to open file selection dialog.', 'error');
        }
    }, [showNotification]); // Добавил showNotification как зависимость


    // --- Стили ---
    const pageContainerStyle = { animation: 'fadeIn 0.5s ease-out' };
    const mainProcessingStyle = { backgroundColor: '#1e1e1e', borderRadius: '16px', boxShadow: '0 8px 20px rgba(0,0,0,0.2)', padding: '24px', border: '1px solid #333', marginTop: '20px' };

    // --- Рендеринг ---
    return (
        <div style={pageContainerStyle}>
             {/* Заголовок и описание для этой страницы */}
            <h1>Weight estimation</h1>
            <p>Upload an image to get an estimate of the animal's weight.</p>

            <div style={mainProcessingStyle}>
                 <ImageUploader
                    selectedFile={selectedFile} isProcessing={isProcessing} isDragging={isDragging}
                    handleSubmit={handleSubmit} handleFileChange={handleFileChange}
                    handleDragEnter={handleDragEnter} handleDragLeave={handleDragLeave}
                    handleDragOver={handleDragOver} handleDrop={handleDrop}
                    handleUploadAreaClick={handleUploadAreaClick}
                    submitButtonText={`Запустить ${TAB_NAMES[pageProcessType]}`} // <--- Текст кнопки
                />
                {processedImage && <ResultViewer processedImage={processedImage} />}
                {isProcessing && <div style={{ textAlign: 'center', margin: '20px 0', color: '#bb86fc' }}>Processing...</div>}
            </div>
        </div>
    );
}
export default WeightPage;