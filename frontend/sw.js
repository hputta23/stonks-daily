const CACHE_NAME = 'stonks-daily-v54';
const ASSETS_TO_CACHE = [
    '/',
    '/index.html',
    '/static/style.css?v=29',
    '/static/app.js?v=53',
    '/static/manifest.json'
];

self.addEventListener('install', event => {
    // Force new service worker to activate immediately
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(ASSETS_TO_CACHE))
    );
});

self.addEventListener('activate', event => {
    const cacheWhitelist = [CACHE_NAME];
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheWhitelist.indexOf(cacheName) === -1) {
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => clients.claim())
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});
