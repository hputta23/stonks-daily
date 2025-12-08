const CACHE_NAME = 'stonks-daily-v16';
const urlsToCache = [
    '/',
    '/static/style.css?v=12',
    '/static/app.js?v=16',
    '/static/manifest.json'
];

self.addEventListener('install', event => {
    // Force new service worker to activate immediately
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('activate', event => {
    // Claim clients immediately so the new SW controls the page
    event.waitUntil(clients.claim());
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});
