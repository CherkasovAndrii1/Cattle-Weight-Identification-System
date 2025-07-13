// src/components/ResultViewer/ResultViewer.jsx
import React from 'react';
import { TAB_NAMES, TAB_KEYS } from '../../constants/tabNames';

const ResultViewer = ({ processedImage }) => {
    if (!processedImage) {
        return null;
    }

    const tabName = TAB_NAMES[processedImage.type] || processedImage.type;
    const isKeypointsTab = processedImage.type === TAB_KEYS.KEYPOINTS;
    const isWeightTab = processedImage.type === TAB_KEYS.WEIGHT;

    // --- Стили (лучше вынести в CSS) ---
    const containerStyle = { border: '1px solid #333', borderRadius: '12px', padding: '20px', backgroundColor: '#252525', animation: 'fadeIn 0.5s, slideUp 0.5s', boxShadow: '0 6px 16px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05)', marginTop: '24px' };
    const titleStyle = { fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#e0e0e0', position: 'relative', display: 'inline-block' };
    const titleUnderlineStyle = { position: 'absolute', bottom: '-4px', left: '0', width: '30px', height: '2px', background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', borderRadius: '2px' };
    const gridStyle = { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', marginBottom: '20px' };
    const imageCardStyle = { borderRadius: '8px', overflow: 'hidden', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)', backgroundColor: '#1a1a1a' };
    const imageHeaderStyle = { fontSize: '14px', color: '#aaa', margin: '0', padding: '10px 12px', backgroundColor: 'rgba(0, 0, 0, 0.2)', borderBottom: '1px solid #333' };
    const imageContainerStyle = { padding: '10px', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' };
    const imgStyle = { maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: '4px' };
    const infoStyle = { backgroundColor: 'rgba(0, 0, 0, 0.15)', padding: '12px 16px', borderRadius: '8px', fontSize: '14px', color: '#aaa', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.1)', gap: '10px' };
    const infoTextStyle = { margin: 0 };
    
    // Стили для отображения веса
    const weightResultStyle = { 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100%',
        padding: '20px'
    };
    const weightValueStyle = {
        fontSize: '32px',
        fontWeight: 'bold',
        color: '#ffffff',
        margin: '10px 0',
        textShadow: '0 2px 4px rgba(0,0,0,0.3)'
    };
    const weightUnitStyle = {
        fontSize: '18px',
        color: '#aaaaaa',
        marginTop: '5px'
    };

    return (
        <div style={containerStyle}>
            <h3 style={titleStyle}>
                Processing result
                <span style={titleUnderlineStyle}></span>
            </h3>
            <div style={gridStyle}>
                {/* Исходное изображение */}
                <div style={imageCardStyle}>
                    <p style={imageHeaderStyle}>Original: {processedImage.filename || ''}</p>
                    <div style={imageContainerStyle}>
                        <img src={processedImage.originalImage} alt="Original" style={imgStyle} />
                    </div>
                </div>
                
                {/* Результат обработки */}
                <div style={imageCardStyle}>
                    <p style={imageHeaderStyle}>Result ({tabName}):</p>
                    <div style={imageContainerStyle}>
                        {isWeightTab ? (
                            // Отображаем информацию о весе
                            <div style={weightResultStyle}>
                                {processedImage.estimatedWeightKg !== null ? (
                                    <>
                                        <div style={weightValueStyle}>
                                            {processedImage.estimatedWeightKg.toFixed(1)}
                                        </div>
                                        <div style={weightUnitStyle}>kilograms</div>
                                    </>
                                ) : (
                                    <div>Unable to determine weight</div>
                                )}
                            </div>
                        ) : (
                            // Для остальных типов показываем обработанное изображение
                            <img
                                src={processedImage.processedImage}
                                alt={`Result ${tabName}`}
                                style={imgStyle}
                                onError={(e) => {
                                    console.error("Error loading processed image:", e);
                                    e.target.alt = "Failed to load result";
                                }}
                            />
                        )}
                    </div>
                </div>
            </div>
            
            {/* Информация об обработке */}
            <div style={infoStyle}>
                <p style={infoTextStyle}>
                    <span role="img" aria-label="clock" style={{ marginRight: '6px' }}>⏱️</span>
                    Time: {processedImage.processingTime}
                </p>
                <p style={infoTextStyle}>
                    <span role="img" aria-label="calendar" style={{ marginRight: '6px' }}>📅</span>
                    Date: {processedImage.timestamp}
                </p>
                
                {/* Информация о ключевых точках */}
                {isKeypointsTab && processedImage.keypoints && (
                    <p style={infoTextStyle}>
                        <span role="img" aria-label="pinpoint" style={{ marginRight: '6px' }}>📍</span>
                        Points found: {processedImage.keypoints.length}
                    </p>
                )}
                
                {/* Информация о статистике сегментации */}
                {processedImage.stats?.cattle_percentage !== undefined && (
                    <p style={infoTextStyle}>
                        <span role="img" aria-label="pie chart" style={{ marginRight: '6px' }}>📊</span>
                        Area (livestock): {processedImage.stats.cattle_percentage}%
                    </p>
                )}
                
                {/* Информация о весе */}
                {isWeightTab && processedImage.estimatedWeightKg !== null && (
                    <p style={infoTextStyle}>
                        <span role="img" aria-label="weight" style={{ marginRight: '6px' }}>⚖️</span>
                        Estimated weight: {processedImage.estimatedWeightKg.toFixed(1)} kilograms
                    </p>
                )}
            </div>
        </div>
    );
};

export default ResultViewer;