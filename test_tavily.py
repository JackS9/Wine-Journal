import os
import unittest
from tavily import TavilyClient

class TestTavilyAPI(unittest.TestCase):

    def setUp(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key)

    def test_search(self):
        query = "Château Margaux 2015"
        results = self.client.search(query=query, search_depth="basic")
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

        for result in results['results']:
            print(result)

if __name__ == "__main__":
    unittest.main()
