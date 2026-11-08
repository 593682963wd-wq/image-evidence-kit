import unittest,tempfile,json
from pathlib import Path
from core import run

class Tests(unittest.TestCase):
 def test_image(self):
  r=run({'synthetic':True});self.assertIn('verdict',r);self.assertNotIn('ml_detector',r['meta']['module_status'])
 def test_empty(self):
  with self.assertRaises(KeyError):run({})
 def test_missing(self):
  with self.assertRaises(ValueError):run({'path':'missing.png'})
 def test_invalid_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.png';p.write_text('not image')
   with self.assertRaises(OSError):run({'path':str(p)})

if __name__=="__main__":unittest.main()
