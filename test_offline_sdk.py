"""作者：Wangyuan；日期：2026-10-29。职责：可选C2PA SDK必须使用显式离线配置。"""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from imagecheck.c2pa_check import _try_official_c2pa


class OfflineSdkTests(unittest.TestCase):
    def test_offline_context_is_required(self):
        captured = {}
        class Context:
            @classmethod
            def from_dict(cls, settings): captured.update(settings); return cls()
            def close(self): pass
        class Reader:
            def __init__(self, path, *, context): self.context = context
            def json(self): return json.dumps({'manifests': {}})
            def close(self): pass
        with patch.dict('sys.modules', {'c2pa':types.SimpleNamespace(Context=Context, Reader=Reader)}):
            self.assertTrue(_try_official_c2pa(Path('synthetic.jpg'))[0])
        self.assertEqual(captured['verify'], {'remote_manifest_fetch':False,'ocsp_fetch':False})
        with patch.dict('sys.modules', {'c2pa':types.SimpleNamespace(Reader=Reader)}):
            self.assertFalse(_try_official_c2pa(Path('synthetic.jpg'))[0])

    def test_unsigned_synthetic_image_with_installed_sdk(self):
        try:
            import c2pa
        except ImportError:
            self.skipTest('可选SDK未安装；离线配置合同仍由独立测试覆盖')
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.jpg'
            Image.new('RGB',(32,32),'white').save(path)
            context = c2pa.Context.from_dict({'verify':{'remote_manifest_fetch':False,'ocsp_fetch':False}})
            try:
                with self.assertRaises(c2pa.C2paError):
                    c2pa.Reader(str(path), context=context)
            finally:
                context.close()
            self.assertEqual(_try_official_c2pa(path), (False,None))


if __name__ == '__main__': unittest.main()
