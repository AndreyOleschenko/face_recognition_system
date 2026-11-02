import unittest
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import app, init_database, get_db_connection

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        self.client = app.test_client()
        with app.app_context():
            init_database()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])

    def test_index(self):
        rv = self.client.get('/')
        self.assertEqual(rv.status_code, 200)

    def test_add_person_page(self):
        rv = self.client.get('/add_person')
        self.assertEqual(rv.status_code, 200)

    # Добавить больше тестов...

if __name__ == '__main__':
    unittest.main()