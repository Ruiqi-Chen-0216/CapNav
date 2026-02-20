// IndexedDB utility for storing and retrieving model files

const DB_NAME = 'CapNavAnnotator';
const DB_VERSION = 1;
const STORE_NAME = 'files';

/**
 * Open the IndexedDB database
 */
export function openDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
    });
}

/**
 * Store a file in IndexedDB
 * @param {File} file - The file to store
 * @param {string} key - The key to store the file under (default: 'modelFile')
 */
export async function storeFile(file, key = 'modelFile') {
    try {
        const db = await openDatabase();
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);

        return new Promise((resolve, reject) => {
            const request = store.put(file, key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    } catch (error) {
        throw new Error(`Failed to store file: ${error.message}`);
    }
}

/**
 * Retrieve a file from IndexedDB
 * @param {string} key - The key of the file to retrieve (default: 'modelFile')
 */
export async function getFile(key = 'modelFile') {
    try {
        const db = await openDatabase();
        const transaction = db.transaction([STORE_NAME], 'readonly');
        const store = transaction.objectStore(STORE_NAME);

        return new Promise((resolve, reject) => {
            const request = store.get(key);
            request.onsuccess = () => {
                if (request.result) {
                    resolve(request.result);
                } else {
                    reject(new Error('No file found in database'));
                }
            };
            request.onerror = () => reject(request.error);
        });
    } catch (error) {
        throw new Error(`Failed to retrieve file: ${error.message}`);
    }
}

/**
 * Delete a file from IndexedDB
 * @param {string} key - The key of the file to delete (default: 'modelFile')
 */
export async function deleteFile(key = 'modelFile') {
    try {
        const db = await openDatabase();
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);

        return new Promise((resolve, reject) => {
            const request = store.delete(key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    } catch (error) {
        throw new Error(`Failed to delete file: ${error.message}`);
    }
}

/**
 * Clear all files from IndexedDB
 */
export async function clearAllFiles() {
    try {
        const db = await openDatabase();
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);

        return new Promise((resolve, reject) => {
            const request = store.clear();
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    } catch (error) {
        throw new Error(`Failed to clear files: ${error.message}`);
    }
}
