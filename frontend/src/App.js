// // src/App.jsx
// import React, { useState, useCallback } from 'react';
// // Импортируем компоненты
// import Header from './components/Header';
// import TabMenu from './components/TabMenu';
// import ImageUploader from './components/ImageUploader';
// import ResultViewer from './components/ResultViewer';
// import HistoryPanel from './components/HistoryPanel';
// import NotificationSystem from './components/NotificationSystem';
// import Footer from './components/Footer';
// // Импортируем хуки, сервисы, константы
// import { useNotification } from './hooks/useNotification';
// import { processImageAPI } from './services/api';
// import { TAB_KEYS, TAB_NAMES } from './constants/tabNames';
// // Импортируем стили (предполагается, что они есть)
// import './styles/global.css';
// import './styles/animations.css'; // Убедитесь, что анимации определены здесь

// function App() {
//     // Основное состояние приложения
//     const [activeTab, setActiveTab] = useState(TAB_KEYS.SEGMENTATION);
//     const [selectedFile, setSelectedFile] = useState(null);
//     const [processedImage, setProcessedImage] = useState(null); // Хранит полный объект результата
//     const [isProcessing, setIsProcessing] = useState(false);
//     const [history, setHistory] = useState([]);
//     const [showHistory, setShowHistory] = useState(false);
//     const [isDragging, setIsDragging] = useState(false);

//     // Используем хук уведомлений
//     const { notification, showNotification } = useNotification();

//     // --- Обработчики ---

//     // Обработчик API вызова
//     const handleProcessImage = useCallback(async (file, processType) => {
//         if (!file) return;
//         setIsProcessing(true);
//         setProcessedImage(null); // Очищаем предыдущий результат

//         // Читаем оригинал для отображения
//         const reader = new FileReader();
//         reader.readAsDataURL(file);
//         reader.onload = async (e) => {
//             const originalImageBase64 = e.target.result;
//             try {
//                 // Вызываем API сервис
//                 const apiResult = await processImageAPI(file, processType);

//                 // Формируем полный объект результата для state
//                  const result = {
//                     id: Date.now(),
//                     type: processType,
//                     originalImage: originalImageBase64,
//                     processedImage: apiResult.image_url, // Используем image_url
//                     timestamp: new Date().toLocaleString(),
//                     processingTime: apiResult.processing_time || 'N/A',
//                     stats: apiResult.stats || null,
//                     filename: file.name
//                 };

//                 setProcessedImage(result);
//                 setHistory(prev => [result, ...prev]); // Добавляем в начало истории
//                 showNotification(`Обработка "${TAB_NAMES[processType]}" завершена!`);

//             } catch (error) {
//                  console.error("Ошибка при обработке изображения в App:", error);
//                  showNotification(error.message || 'Произошла ошибка при обработке');
//                  setProcessedImage(null); // Сбрасываем результат при ошибке
//             } finally {
//                 setIsProcessing(false);
//             }
//         };
//         reader.onerror = (error) => {
//             console.error("Ошибка чтения файла:", error);
//             showNotification('Не удалось прочитать файл');
//             setIsProcessing(false); // Важно сбросить флаг
//         };
//     }, [showNotification]); // Добавили showNotification в зависимости useCallback

//     // Обработчик отправки формы (вызывает handleProcessImage)
//     const handleSubmit = useCallback((e) => {
//         e.preventDefault();
//         if (selectedFile) {
//             handleProcessImage(selectedFile, activeTab);
//         }
//     }, [selectedFile, activeTab, handleProcessImage]);

//      // Обработчик выбора файла
//      const handleFileChange = useCallback((e) => {
//         if (e.target.files && e.target.files[0]) {
//              const file = e.target.files[0];
//              if (file.type.startsWith('image/')) {
//                  setSelectedFile(file);
//                  setProcessedImage(null);
//                  showNotification(`Файл "${file.name}" успешно выбран`);
//              } else {
//                   showNotification('Пожалуйста, выберите файл изображения.');
//              }
//         }
//     }, [showNotification]);

//     // Обработчик загрузки из истории
//     const loadHistoryItem = useCallback((item) => {
//         setActiveTab(item.type);
//         setProcessedImage(item);
//         setSelectedFile(null); // Сбрасываем файл, т.к. грузим из истории
//         setShowHistory(false);
//     }, []);

//     // Обработчики Drag & Drop
//     const handleDragEnter = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }, []);
//     const handleDragLeave = useCallback((e) => { e.preventDefault(); e.stopPropagation(); if (e.currentTarget.contains(e.relatedTarget)) return; setIsDragging(false); }, []);
//     const handleDragOver = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }, []);
//     const handleDrop = useCallback((e) => {
//         e.preventDefault(); e.stopPropagation(); setIsDragging(false);
//         if (e.dataTransfer.files && e.dataTransfer.files[0]) {
//             const file = e.dataTransfer.files[0];
//             if (file.type.startsWith('image/')) {
//                 setSelectedFile(file); setProcessedImage(null);
//                 showNotification(`Файл "${file.name}" успешно выбран`);
//             } else { showNotification('Пожалуйста, выберите файл изображения.'); }
//         }
//     }, [showNotification]);

//     // Клик по области загрузки
//     const handleUploadAreaClick = useCallback(() => {
//         const fileInput = document.getElementById('file-upload');
//         if (fileInput) { fileInput.click(); }
//     }, []);

//     // Обработчик смены активной вкладки
//     const handleSetActiveTab = useCallback((tabKey) => {
//          setActiveTab(tabKey);
//          setShowHistory(false); // Скрываем историю при смене основной вкладки
//          setProcessedImage(null); // Очищаем результат
//          setSelectedFile(null);   // Очищаем выбранный файл
//     }, []);


//     // --- Рендеринг ---
//     // Стили контейнера App лучше вынести в CSS
//     const appContainerStyle = { display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: '#121212', color: '#e0e0e0', fontFamily: '"Segoe UI", Roboto, Arial, sans-serif' };
//     const mainStyle = { flex: '1', maxWidth: '1200px', width: '100%', margin: '24px auto', padding: '0 16px' };
//     const mainContentContainerStyle = { backgroundColor: '#1e1e1e', borderRadius: '16px', boxShadow: '0 8px 20px rgba(0,0,0,0.2)', padding: '24px', border: '1px solid #333', animation: 'fadeIn 0.5s ease-out' };


//     return (
//         <div style={appContainerStyle}>
//             <NotificationSystem notification={notification} />
//             <Header />
//             <TabMenu
//                 activeTab={activeTab}
//                 setActiveTab={handleSetActiveTab} // Передаем новый обработчик
//                 showHistory={showHistory}
//                 setShowHistory={setShowHistory}
//             />

//             <main style={mainStyle}>
//                 {showHistory ? (
//                     <HistoryPanel history={history} loadHistoryItem={loadHistoryItem} />
//                 ) : (
//                     <div style={mainContentContainerStyle}>
//                         <ImageUploader
//                             activeTab={activeTab}
//                             selectedFile={selectedFile}
//                             isProcessing={isProcessing}
//                             isDragging={isDragging}
//                             handleSubmit={handleSubmit}
//                             handleFileChange={handleFileChange}
//                             handleDragEnter={handleDragEnter}
//                             handleDragLeave={handleDragLeave}
//                             handleDragOver={handleDragOver}
//                             handleDrop={handleDrop}
//                             handleUploadAreaClick={handleUploadAreaClick}
//                         />
//                         {/* Отображаем результат только если он есть и соответствует активной вкладке */}
//                         {processedImage && processedImage.type === activeTab && (
//                              <ResultViewer processedImage={processedImage} />
//                         )}
//                     </div>
//                 )}
//             </main>

//             <Footer />
//         </div>
//     );
// }

// export default App;