import logging
import os
import json
import lmdb
from typing import Optional, Dict, List, Iterator
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")

class CacheInteractionTool:
    def __init__(self, data_dir: str, block_set_dir: Optional[str] = None):
        """
        Initialize the CacheInteractionTool with the specified dataset directory.
        Args:
            data_dir: Path to the directory containing dataset files.
            block_set_dir: Optional path to the directory containing block set files.
        """
        logger.info(f"Initializing CacheInteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir
        self.block_set_dir = block_set_dir

        # Set up LMDB environments for caching
        self.env_dir = os.path.join(data_dir, "lmdb_cache")
        os.makedirs(self.env_dir, exist_ok=True)

        self.user_env = lmdb.open(os.path.join(self.env_dir, "users"), map_size=2 * 1024 * 1024 * 1024)
        self.item_env = lmdb.open(os.path.join(self.env_dir, "items"), map_size=2 * 1024 * 1024 * 1024)
        self.review_env = lmdb.open(os.path.join(self.env_dir, "reviews"), map_size=8 * 1024 * 1024 * 1024)

        # Load block set data if provided
        self.block_set_items = []
        self.block_set_pairs = set()  # Store (user_id, item_id) pairs
        if self.block_set_dir:
            logger.info(f"Loading block set data from {self.block_set_dir}")
            self.block_set_items = self._load_block_set()
            self.block_set_pairs = {(item['user_id'], item['item_id']) for item in self.block_set_items}

        # Filter reviews based on block set
        filtered_reviews = []
        with self.review_env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                review = json.loads(value)
                if (review['user_id'], review['item_id']) not in self.block_set_pairs:
                    filtered_reviews.append(review)
        
        logger.info(f"Filtered out {txn.stat()['entries'] - len(filtered_reviews)} reviews based on block set")
        self.review_env = filtered_reviews

        self._initialize_db()

    def _load_block_set(self) -> List[Dict[str, str]]:
        """Load block set files and return a list of user-item pairs."""
        block_set_data = []
        task_dir = os.path.join(self.block_set_dir, 'tasks')
        groundtruth_dir = os.path.join(self.block_set_dir, 'groundtruth')
        
        for filename in os.listdir(task_dir):
            if filename.startswith('task_') and filename.endswith('.json'):
                task_file_path = os.path.join(task_dir, filename)
                with open(task_file_path, 'r', encoding='utf-8') as task_file:
                    task_data = json.load(task_file)
                    if task_data["type"] == "user_behavior_simulation":
                        block_set_data.append({'user_id': task_data['user_id'], 'item_id': task_data['item_id']})
                    else:
                        groundtruth_filename = filename.replace('task_', 'groundtruth_')
                        groundtruth_file_path = os.path.join(groundtruth_dir, groundtruth_filename)
                        with open(groundtruth_file_path, 'r', encoding='utf-8') as groundtruth_file:
                            groundtruth_data = json.load(groundtruth_file)
                            for item in task_data['candidate_list']:
                                if item == groundtruth_data['ground truth']:
                                    block_set_data.append({'user_id': task_data['user_id'], 'item_id': item})
        return block_set_data

    def _initialize_db(self):
        """Initialize the LMDB databases with data if they are empty."""
        # Initialize users
        with self.user_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for user in tqdm(self._iter_file('user.json')):
                        cursor.put(
                            user['user_id'].encode(),
                            json.dumps(user).encode()
                        )

        # Initialize items
        with self.item_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for item in tqdm(self._iter_file('item.json')):
                        cursor.put(
                            item['item_id'].encode(),
                            json.dumps(item).encode()
                        )

        # Initialize reviews and their indices
        with self.review_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                for review in tqdm(self._iter_file('review.json')):
                    # Store the review
                    txn.put(
                        review['review_id'].encode(),
                        json.dumps(review).encode()
                    )

                    # Update item reviews index (store only review_ids)
                    item_review_ids = json.loads(txn.get(f"item_{review['item_id']}".encode()) or '[]')
                    item_review_ids.append(review['review_id'])
                    txn.put(
                        f"item_{review['item_id']}".encode(),
                        json.dumps(item_review_ids).encode()
                    )

                    # Update user reviews index (store only review_ids)
                    user_review_ids = json.loads(txn.get(f"user_{review['user_id']}".encode()) or '[]')
                    user_review_ids.append(review['review_id'])
                    txn.put(
                        f"user_{review['user_id']}".encode(),
                        json.dumps(user_review_ids).encode()
                    )

    def _iter_file(self, filename: str) -> Iterator[Dict]:
        """Iterate through file line by line."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""
        with self.user_env.begin() as txn:
            user_data = txn.get(user_id.encode())
            if user_data:
                return json.loads(user_data)
        return None

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        if not item_id:
            return None

        with self.item_env.begin() as txn:
            item_data = txn.get(item_id.encode())
            if item_data:
                return json.loads(item_data)
        return None

    def get_reviews(
            self,
            item_id: Optional[str] = None,
            user_id: Optional[str] = None,
            review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            with self.review_env.begin() as txn:
                review_data = txn.get(review_id.encode())
                if review_data:
                    return [json.loads(review_data)]
            return []

        with self.review_env.begin() as txn:
            if item_id:
                review_ids = json.loads(txn.get(f"item_{item_id}".encode()) or '[]')
            elif user_id:
                review_ids = json.loads(txn.get(f"user_{user_id}".encode()) or '[]')
            else:
                return []

            # Fetch complete review data for each review_id
            reviews = []
            for rid in review_ids:
                review_data = txn.get(rid.encode())
                if review_data:
                    reviews.append(json.loads(review_data))
            return reviews

    def __del__(self):
        """Cleanup LMDB environments on object destruction."""
        self.user_env.close()
        self.item_env.close()
        self.review_env.close()