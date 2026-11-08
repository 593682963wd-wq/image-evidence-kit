import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from imagecheck.classifier import _features
from imagecheck.exif_check import _read_exif
from imagecheck.fft_check import analyze_fft
from imagecheck.ela_check import analyze_ela
from imagecheck.prnu_check import analyze_prnu
from imagecheck.evidence import Direction

class ApiRegressions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'synthetic.png'
    def test_entropy_is_probability_entropy(self):
        gray=np.tile(np.arange(256,dtype=np.uint8),(256,1))
        Image.fromarray(gray).save(self.path)
        self.assertGreater(_features(self.path)['color_entropy'],5.5)
    def test_nested_exif_camera_fields(self):
        exif=Image.Exif();exif[271]='Synthetic';exif[272]='Camera'
        exif[34665]={33434:0.01,34855:100,33437:2.8,36867:'2026:01:01 12:00:00'}
        path=self.path.with_suffix('.jpg');Image.new('RGB',(64,64)).save(path,exif=exif)
        result=_read_exif(path)
        self.assertTrue({'ExposureTime','ISOSpeedRatings','FNumber','DateTimeOriginal'} <= result.keys())
    def test_zero_fft_energy_is_neutral(self):
        Image.new('RGB',(128,128),'black').save(self.path)
        evs,info=analyze_fft(self.path)
        self.assertTrue(evs)
        self.assertTrue(all(e.direction==Direction.NEUTRAL for e in evs))
        self.assertEqual(info['hf_ratio'],0)
        self.assertTrue(np.isfinite(info['log_spectrum']).all())
    def test_small_ela_valid(self):
        Image.new('RGB',(16,16),'white').save(self.path)
        evs,info=analyze_ela(self.path)
        self.assertTrue(info)
        self.assertNotIn('error',[e.name for e in evs])
    def test_tall_images_keep_nonzero_size(self):
        Image.new('RGB',(1,2048),'gray').save(self.path)
        for analyze in (analyze_ela,analyze_prnu):
            with self.subTest(analyze=analyze.__name__):
                evs,info=analyze(self.path)
                self.assertTrue(info)
                self.assertNotIn('error',[e.name for e in evs])
