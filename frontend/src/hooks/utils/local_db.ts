// ... (alongside saveFileToIndexedDB)

export function getFileFromIndexedDB(fileName: string): Promise<File> {
  return new Promise((resolve, reject) => {
    const dbName = "OriginalFileDB";
    const storeName = "originalFiles";
    const request = indexedDB.open(dbName, 1);

    request.onerror = (event) => reject("IndexedDB error");

    request.onsuccess = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      const transaction = db.transaction(storeName, "readonly");
      const store = transaction.objectStore(storeName);
      const getRequest = store.get(fileName);

      getRequest.onsuccess = () => {
        if (getRequest.result) {
          resolve(getRequest.result);
        } else {
          reject(`File '${fileName}' not found in IndexedDB.`);
        }
      };
      getRequest.onerror = () => reject("Error getting file from IndexedDB");
      transaction.oncomplete = () => db.close();
    };
  });
}