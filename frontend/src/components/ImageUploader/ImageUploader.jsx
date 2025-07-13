// src/components/ImageUploader/ImageUploader.jsx
import React from 'react';
import { TAB_NAMES } from '../../constants/tabNames'; // Импортируем имена вкладок

const ImageUploader = (props) => {
    const {
        activeTab,
        selectedFile,
        isProcessing,
        isDragging,
        handleSubmit,
        handleFileChange,
        handleDragEnter,
        handleDragLeave,
        handleDragOver,
        handleDrop,
        handleUploadAreaClick,
    } = props;

    const tabName = TAB_NAMES[activeTab] || activeTab;

    // Стили лучше вынести
    const containerStyle = { /* ... */ };
    const titleStyle = { /* ... */ };
    const titleUnderlineStyle = { /* ... */ };
    const formStyle = { marginBottom: '24px' };
    const labelStyle = { display: 'block', marginBottom: '12px', color: '#cccccc', fontWeight: '500', fontSize: '15px' };
    const dropAreaBaseStyle = { border: `2px dashed #444`, borderRadius: '12px', padding: '28px', textAlign: 'center', backgroundColor: '#252525', transition: 'all 0.3s ease', cursor: 'pointer', position: 'relative' };
    const dropAreaDraggingStyle = { ...dropAreaBaseStyle, border: `2px dashed #3b82f6`, backgroundColor: 'rgba(59, 130, 246, 0.1)' };
    const dropContentStyle = { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' };
    const iconContainerStyle = { width: '64px', height: '64px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px', transition: 'transform 0.3s' };
    const iconDraggingStyle = { ...iconContainerStyle, animation: 'pulse 1.5s infinite' };
    const fileNameStyle = { margin: '0 0 8px', color: '#aaa', fontSize: '16px', fontWeight: '500' };
    const fileHintStyle = { margin: 0, color: '#777', fontSize: '13px' };
    const submitButtonStyle = { width: '100%', padding: '14px', color: 'white', border: 'none', borderRadius: '12px', fontSize: '16px', fontWeight: '600', transition: 'all 0.3s', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden' };
    const submitButtonActiveStyle = { ...submitButtonStyle, background: 'linear-gradient(135deg, #2563eb, #4f46e5)', cursor: 'pointer', opacity: 1, boxShadow: '0 4px 12px rgba(37, 99, 235, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)'};
    const submitButtonDisabledStyle = { ...submitButtonStyle, background: '#555', cursor: 'not-allowed', opacity: 0.7, boxShadow: 'none'};
    const spinnerStyle = { marginRight: '8px', animation: 'spin 1s linear infinite' };
    const shimmerStyle = { position: 'absolute', top: '0', left: '-100%', width: '200%', height: '100%', background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)', animation: 'shimmer 2s infinite' };


    return (
        <div style={containerStyle}>
            <h2 style={titleStyle}>
                {tabName}
                <span style={titleUnderlineStyle}></span>
            </h2>
            <form onSubmit={handleSubmit} style={formStyle}>
                <div style={{ marginBottom: '16px' }}>
                    <label style={labelStyle}>Upload your file:</label>
                    <div
                        onClick={handleUploadAreaClick}
                        onDragEnter={handleDragEnter}
                        onDragLeave={handleDragLeave}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        style={isDragging ? dropAreaDraggingStyle : dropAreaBaseStyle}
                    >
                        <div style={dropContentStyle}>
                            <div style={isDragging ? iconDraggingStyle : iconContainerStyle}>
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="17 8 12 3 7 8"/>
                                    <line x1="12" y1="3" x2="12" y2="15"/>
                                </svg>
                            </div>
                            <p style={fileNameStyle}>
                                {selectedFile ? `Chosen: ${selectedFile.name}` : 'Drag and drop file here or click to upload'}
                            </p>
                            <p style={fileHintStyle}>Supported formats: JPEG, PNG, GIF, WEBP
                            </p>
                        </div>
                        <input
                            id="file-upload" type="file"
                            accept="image/jpeg, image/png, image/gif, image/webp"
                            onChange={handleFileChange}
                            style={{ display: 'none' }}
                        />
                    </div>
                </div>
                <button
                    type="submit"
                    disabled={!selectedFile || isProcessing}
                    style={(!selectedFile || isProcessing) ? submitButtonDisabledStyle : submitButtonActiveStyle}
                    onMouseOver={(e) => { if (!(!selectedFile || isProcessing)) e.currentTarget.style.opacity = '0.9'; }}
                    onMouseOut={(e) => { if (!(!selectedFile || isProcessing)) e.currentTarget.style.opacity = '1'; }}
                >
                    <span style={{ position: 'relative', zIndex: 2, display: 'flex', alignItems: 'center' }}>
                        {isProcessing ? (
                            <>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style={spinnerStyle}>
                                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeOpacity="0.25"/>
                                    <path d="M12 2a10 10 0 0 0-10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round"/>
                                </svg>
                                Processing...
                            </>
                        ) : 'Process'}
                    </span>
                    {!isProcessing && selectedFile && <span style={shimmerStyle}></span>}
                </button>
            </form>
        </div>
    );
};

export default ImageUploader;