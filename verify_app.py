import os
import sys
sys.path.insert(0, os.getcwd())
from app import app
print('import_ok', app is not None)
print('routes', [rule.rule for rule in app.url_map.iter_rules()])
