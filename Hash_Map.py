class HashMap:

    # Create empty bucket list of given size to start
    def __init__(self, size):
        self.size = size
        self.hash_map = self.create_buckets()

    def create_buckets(self):
        return [[] for _ in range(self.size)]


    def put(self, key, val):

        hashed_key = hash(key) % self.size

        bucket = self.hash_map[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    def get_val(self, key):

        hashed_key = hash(key) % self.size

        bucket = self.hash_map[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            if record_key == key:
                found_key = True
                break

        if found_key:
            return record_val
        else:
            return "Answer missing"

