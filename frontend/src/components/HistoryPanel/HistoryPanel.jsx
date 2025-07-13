// src/components/HistoryPanel/HistoryPanel.jsx
import React from 'react';
import { TAB_NAMES } from '../../constants/tabNames';
import { Typography } from '@mui/material'; // Импортируем Typography для заглушки

const HistoryPanel = ({ history, loadHistoryItem }) => {
    // --- Стили (оставляем как есть) ---
    const containerStyle = { backgroundColor: '#1e1e1e', borderRadius: '16px', boxShadow: '0 8px 20px rgba(0,0,0,0.2)', padding: '24px', border: '1px solid #333', animation: 'fadeIn 0.5s ease-out' };
    const titleStyle = { fontSize: '22px', fontWeight: '600', marginBottom: '20px', color: '#e0e0e0', position: 'relative', display: 'inline-block' };
    const titleUnderlineStyle = { position: 'absolute', bottom: '-4px', left: '0', width: '40px', height: '3px', background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', borderRadius: '2px' };
    const emptyHistoryStyle = { textAlign: 'center', padding: '40px 0', color: '#aaa' };
    const emptyIconStyle = { margin: '0 auto 16px', display: 'block' };
    const emptyTextStyle = { fontSize: '16px' };
    const emptyHintStyle = { fontSize: '14px', color: '#777', maxWidth: '400px', margin: '8px auto' };
    const historyListStyle = { display: 'flex', flexDirection: 'column', gap: '16px' };
    const historyItemStyle = { border: '1px solid #333', borderRadius: '12px', padding: '16px', backgroundColor: '#252525', transition: 'all 0.3s', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05)', cursor: 'pointer' };
    const itemHeaderStyle = { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' };
    const itemTitleStyle = { fontSize: '16px', fontWeight: '600', margin: '0 0 4px', color: '#e0e0e0', wordBreak: 'break-all' }; // Добавил wordBreak
    const itemTimestampStyle = { fontSize: '13px', color: '#aaa', margin: 0 };
    const thumbsGridStyle = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', borderRadius: '8px', overflow: 'hidden', marginTop: '10px' }; // Добавил marginTop
    // Стиль для контейнера превью (где будет или картинка, или заглушка)
    const thumbContainerStyle = {
        borderRadius: '6px', overflow: 'hidden',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
        backgroundColor: '#1a1a1a', height: '120px', // Увеличил высоту для лучшего вида
        display:'flex', alignItems:'center', justifyContent:'center',
        position: 'relative' // Для подписи
    };
    const thumbImgStyle = { display: 'block', width: '100%', height: '100%', objectFit: 'contain', transition: 'transform 0.3s' };
    // Стиль для заглушки "Нет изображения"
    const noImageStyle = { color: '#666', fontSize: '12px', textAlign: 'center' };
    // Стиль для подписи под картинкой
    const thumbCaptionStyle = {
        position: 'absolute', bottom: '5px', left: '5px', right: '5px',
        backgroundColor: 'rgba(0, 0, 0, 0.6)', color: 'white',
        fontSize: '11px', padding: '2px 4px', borderRadius: '3px',
        textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
    };


    const getTabName = (tabKey) => TAB_NAMES[tabKey] || tabKey;

    return (
        <div style={containerStyle}>
            <h2 style={titleStyle}>
                Request history
                <span style={titleUnderlineStyle}></span>
            </h2>

            {history.length === 0 ? (
                <div style={emptyHistoryStyle}>
                    {/* SVG и текст без изменений */}
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#666" strokeWidth="1.5" style={emptyIconStyle}><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"></path><path d="M12 6v6l4 2"></path></svg>
                    <p style={emptyTextStyle}>The search history is empty.</p>
                    <p style={emptyHintStyle}>Process the image to see it in the history.</p>
                </div>
            ) : (
                <div style={historyListStyle}>
                    {history.map((item, index) => (
                        <div
                            key={item.id}
                            style={{ ...historyItemStyle, animation: `fadeIn 0.5s ${index * 0.05}s both` }}
                            onClick={() => loadHistoryItem(item)}
                            onMouseOver={(e) => e.currentTarget.style.borderColor = '#555'}
                            onMouseOut={(e) => e.currentTarget.style.borderColor = '#333'}
                        >
                            <div style={itemHeaderStyle}>
                                <div>
                                    <h3 style={itemTitleStyle}>
                                        {getTabName(item.processing_type)} - {item.original_filename ? (item.original_filename.length > 25 ? item.original_filename.substring(0, 22) + '...' : item.original_filename) : 'Без имени'}
                                    </h3>
                                    {/* Используем поле uploaded_at */}
                                    <p style={itemTimestampStyle}>{new Date(item.uploaded_at).toLocaleString('ru-RU')}</p>
                                </div>
                                {/* Можно добавить кнопку удаления или доп. инфо */}
                            </div>

                            {/* <<< НАЧАЛО ИЗМЕНЕНИЙ: Отображение картинок >>> */}
                            <div style={thumbsGridStyle}>
                                {/* Контейнер для Оригинала (пока заглушка) */}
                                <div style={thumbContainerStyle}>
                                    <Typography variant="caption" style={noImageStyle}>
                                        (Original was not saved yet)
                                    </Typography>
                                     <div style={thumbCaptionStyle}>Original</div>
                                </div>

                                {/* Контейнер для Результата */}
                                <div style={thumbContainerStyle}>
                                    {item.result_image_url ? (
                                        <img
                                            src={item.result_image_url} // <-- ИСПОЛЬЗУЕМ ПРАВИЛЬНОЕ ПОЛЕ
                                            alt={`Результат ${getTabName(item.processing_type)}`}
                                            style={thumbImgStyle}
                                            // Добавляем обработчик ошибок загрузки картинки
                                            onError={(e) => {
                                                e.target.onerror = null; // Предотвращаем бесконечный цикл, если и заглушка не загрузится
                                                e.target.style.display = 'none'; // Скрываем сломанный img
                                                // Можно показать текст ошибки внутри контейнера
                                                const parent = e.target.parentNode;
                                                if(parent) {
                                                     const errorText = parent.querySelector('.error-text');
                                                     if(errorText) errorText.style.display = 'block';
                                                }
                                            }}
                                        />
                                    ) : (
                                        // Показываем заглушку, если URL нет
                                        <Typography variant="caption" style={noImageStyle}>
                                            No image
                                        </Typography>
                                    )}
                                     {/* Текст ошибки, который покажется при onError */}
                                     <Typography variant="caption" className="error-text" style={{...noImageStyle, display: 'none', color: '#ff6b6b'}}>Ошибка</Typography>
                                     <div style={thumbCaptionStyle}>Result</div>
                                </div>
                            </div>
                             {}

                             {}
                             {item.processing_type === 'weight' && item.result_data?.estimated_weight_kg && (
                                <Typography variant="body2" sx={{ mt: 1, fontWeight: 'bold' }}>
                                    Weight: {item.result_data.estimated_weight_kg} кг
                                </Typography>
                             )}
                             {/* Добавьте здесь отображение другой информации из item.result_data при необходимости */}

                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default HistoryPanel;