"""
MongoDB database connection and collection management for MAUSAM.
Supports native MongoDB instances and transparent local persistent storage fallback.
"""
import logging
import json
import os
from typing import Dict, Any, List, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING, GEOSPHERE
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from ..config import settings

logger = logging.getLogger("mausam.database")

class InMemoryMongoCollection:
    """Mock/Fallback collection matching essential PyMongo interface."""
    def __init__(self, name: str, parent_db):
        self.name = name
        self.parent_db = parent_db
        self.docs: List[Dict[str, Any]] = []

    def insert_one(self, document: Dict[str, Any]):
        doc_copy = dict(document)
        if "_id" not in doc_copy:
            doc_copy["_id"] = str(doc_copy.get("id") or len(self.docs) + 1)
        self.docs.append(doc_copy)
        self.parent_db.save_fallback()
        class InsertResult:
            inserted_id = doc_copy["_id"]
        return InsertResult()

    def insert_many(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            self.insert_one(doc)

    def find(self, filter_query: Optional[Dict[str, Any]] = None, sort=None, limit: int = 0):
        results = []
        filter_query = filter_query or {}
        for doc in self.docs:
            match = True
            for k, v in filter_query.items():
                if k not in doc or doc[k] != v:
                    match = False
                    break
            if match:
                results.append(dict(doc))
        if sort:
            key, direction = sort[0]
            reverse = (direction == -1 or direction == DESCENDING)
            results.sort(key=lambda x: x.get(key, 0) or 0, reverse=reverse)
        if limit > 0:
            results = results[:limit]
        return results

    def find_one(self, filter_query: Optional[Dict[str, Any]] = None):
        res = self.find(filter_query, limit=1)
        return res[0] if res else None

    def update_one(self, filter_query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        target = self.find_one(filter_query)
        if target:
            if "$set" in update:
                target.update(update["$set"])
            self.parent_db.save_fallback()
            return True
        elif upsert and "$set" in update:
            new_doc = dict(filter_query)
            new_doc.update(update["$set"])
            self.insert_one(new_doc)
            return True
        return False

    def count_documents(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        return len(self.find(filter_query))

    def create_index(self, keys, **kwargs):
        pass


class MongoCollectionWrapper:
    """Wraps PyMongo collection to automatically convert ObjectId to string for seamless JSON serialization."""
    def __init__(self, collection):
        self._col = collection

    def _clean(self, doc):
        if doc and isinstance(doc, dict):
            doc = dict(doc)
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        return doc

    def insert_one(self, doc):
        return self._col.insert_one(doc)

    def insert_many(self, docs):
        return self._col.insert_many(docs)

    def find(self, filter_query=None, sort=None, limit=0):
        filter_query = filter_query or {}
        cursor = self._col.find(filter_query)
        if sort:
            cursor = cursor.sort(sort)
        if limit > 0:
            cursor = cursor.limit(limit)
        return [self._clean(d) for d in cursor]

    def find_one(self, filter_query=None):
        doc = self._col.find_one(filter_query or {})
        return self._clean(doc)

    def update_one(self, filter_query, update, upsert=False):
        return self._col.update_one(filter_query, update, upsert=upsert)

    def count_documents(self, filter_query=None):
        return self._col.count_documents(filter_query or {})

    def create_index(self, keys, **kwargs):
        return self._col.create_index(keys, **kwargs)

    def __getattr__(self, name):
        return getattr(self._col, name)


class DatabaseManager:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.is_connected: bool = False
        import tempfile
        base_dir = tempfile.gettempdir() if (os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")) else os.getcwd()
        self.fallback_file = os.path.join(base_dir, ".data", "local_mongo_store.json")
        self.collections: Dict[str, Any] = {}
        self.connect()

    def connect(self):
        # On Vercel, if MONGODB_URI is pointing to localhost, immediately use local store without timeout
        if os.environ.get("VERCEL") and ("localhost" in settings.MONGODB_URI or "127.0.0.1" in settings.MONGODB_URI):
            logger.info("Serverless environment detected without external MongoDB Atlas URI. Activating embedded local store.")
            self._init_fallback_collections()
            return

        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI} (db: {settings.MONGODB_DB_NAME})...")
            self.client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=1000)
            self.client.admin.command("ping")
            self.db = self.client[settings.MONGODB_DB_NAME]
            self.is_connected = True
            logger.info("Successfully connected to live MongoDB!")
            self._init_mongo_collections()
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            logger.warning(f"Could not connect to MongoDB server ({e}).")
            if settings.ENABLE_MONGO_LOCAL_FALLBACK:
                logger.info("Activating embedded local document store fallback. Zero data loss!")
                self._init_fallback_collections()
            else:
                raise e

    def _init_mongo_collections(self):
        self.forecast_runs = MongoCollectionWrapper(self.db["forecast_runs"])
        self.anomalies = MongoCollectionWrapper(self.db["anomalies"])
        self.downscaled_grids = MongoCollectionWrapper(self.db["downscaled_grids"])
        self.alerts = MongoCollectionWrapper(self.db["alerts"])
        self.subscriptions = MongoCollectionWrapper(self.db["subscriptions"])
        self.audit_logs = MongoCollectionWrapper(self.db["audit_logs"])

    def _init_fallback_collections(self):
        try:
            os.makedirs(os.path.dirname(self.fallback_file), exist_ok=True)
        except OSError:
            pass
        raw_data = {}
        if os.path.exists(self.fallback_file):
            try:
                with open(self.fallback_file, "r") as f:
                    raw_data = json.load(f)
            except Exception:
                raw_data = {}

        col_names = ["forecast_runs", "anomalies", "downscaled_grids", "alerts", "subscriptions", "audit_logs"]
        for name in col_names:
            c = InMemoryMongoCollection(name, self)
            c.docs = raw_data.get(name, [])
            setattr(self, name, c)
            self.collections[name] = c

    def save_fallback(self):
        if not self.is_connected:
            try:
                os.makedirs(os.path.dirname(self.fallback_file), exist_ok=True)
                data = {name: getattr(self, name).docs for name in self.collections}
                with open(self.fallback_file, "w") as f:
                    json.dump(data, f, default=str, indent=2)
            except Exception as e:
                logger.warning(f"Notice: embedded fallback store saved in-memory: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "Live MongoDB" if self.is_connected else "Local Embedded Document Store",
            "connected": self.is_connected,
            "database_name": settings.MONGODB_DB_NAME,
            "uri": settings.MONGODB_URI.split("@")[-1] if "@" in settings.MONGODB_URI else settings.MONGODB_URI,
            "counts": {
                "forecast_runs": self.forecast_runs.count_documents({}),
                "anomalies": self.anomalies.count_documents({}),
                "downscaled_grids": self.downscaled_grids.count_documents({}),
                "alerts": self.alerts.count_documents({}),
                "subscriptions": self.subscriptions.count_documents({})
            }
        }


db_manager = DatabaseManager()
db = db_manager

def get_database():
    return db_manager

def init_db_indexes():
    """Create essential MongoDB indexes for high-speed spatial & temporal queries."""
    if db_manager.is_connected:
        try:
            db_manager.alerts.create_index([("centroid_lat", ASCENDING), ("centroid_lon", ASCENDING)])
            db_manager.alerts.create_index([("severity", ASCENDING), ("active", ASCENDING)])
            db_manager.anomalies.create_index([("anomaly_id", ASCENDING)], unique=True)
            db_manager.anomalies.create_index([("run_id", ASCENDING), ("max_efi", DESCENDING)])
            db_manager.downscaled_grids.create_index([("grid_id", ASCENDING)], unique=True)
            db_manager.downscaled_grids.create_index([("anomaly_id", ASCENDING), ("lead_day", ASCENDING)])
            logger.info("MongoDB database indexes successfully verified and initialized.")
        except Exception as e:
            logger.warning(f"Failed to create some MongoDB indexes: {e}")
