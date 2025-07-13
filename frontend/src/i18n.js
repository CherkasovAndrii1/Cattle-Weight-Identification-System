// src/i18n.js
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import Backend from 'i18next-http-backend'; // Для загрузки переводов

i18n
  // Загрузка переводов с сервера/папки public
  .use(Backend)
  // Определение языка браузера
  .use(LanguageDetector)
  // Передача экземпляра i18n в react-i18next
  .use(initReactI18next)
  // Инициализация i18next
  .init({
    // Языки, которые мы поддерживаем
    supportedLngs: ['ru', 'en'],
    // Язык по умолчанию, если определение не сработало
    fallbackLng: 'ru',
    debug: true, // Включить логирование в консоль (удобно при разработке)
    detection: {
      // Порядок и способы определения языка
      order: ['querystring', 'cookie', 'localStorage', 'sessionStorage', 'navigator', 'htmlTag'],
      // Ключи для поиска в cache (например, localStorage)
      caches: ['localStorage', 'cookie'],
    },
    backend: {
      // Путь, откуда загружать файлы переводов
      // /locales/{{lng}}/translation.json
      // где {{lng}} будет заменен на код языка (ru, en)
      loadPath: '/locales/{{lng}}/translation.json',
    },
    interpolation: {
      escapeValue: false, // React уже защищает от XSS
    },
    react: {
      // Используем Suspense для асинхронной загрузки переводов
      useSuspense: true,
    }
  });

export default i18n;