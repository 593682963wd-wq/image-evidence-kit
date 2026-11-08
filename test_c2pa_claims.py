"""作者：Wangyuan；日期：2026-10-29；职责：C2PA声明不能冒充来源证明。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from imagecheck.c2pa_check import detect_c2pa
from imagecheck.evidence import Direction,Strength

class ClaimTests(unittest.TestCase):
    def test_signer_and_status_never_prove_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic.bin';path.write_bytes(b'c2pa')
            for signer,status in [('Canon',[{'code':'claimSignature.mismatch'}]),('OpenAI',None),('Canon',[])]:
                manifest={'active_manifest':'sample','manifests':{'sample':{'signature_info':{'issuer':signer}}},'validation_status':status}
                with self.subTest(signer=signer,status=status),patch('imagecheck.c2pa_check._try_official_c2pa',return_value=(True,manifest)):
                    result=detect_c2pa(path)[0]
                    self.assertEqual(result.direction,Direction.NEUTRAL)
                    self.assertEqual(result.strength,Strength.INFO)
                    self.assertEqual(result.score,0)
                    self.assertFalse(result.raw['verified'])
