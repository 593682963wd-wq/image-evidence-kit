"""作者：Wangyuan；日期：2026-10-29；职责：离线图像证据，拒绝将统计线索当确证。"""
import tempfile
from pathlib import Path
from PIL import Image
from imagecheck.pipeline import detect

def run(payload):
    if not isinstance(payload,dict):raise TypeError('输入必须为对象')
    if payload.get('synthetic'):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'synthetic.png';Image.new('RGB',(128,128),(80,120,160)).save(path)
            return run({'path':str(path)})
    path=Path(payload['path'])
    if not path.is_file():raise ValueError('必须提供图像文件')
    with Image.open(path) as im:im.verify()
    result=detect(path,use_ml=False)
    result['meta'].pop('image_path',None)
    result['notice']='图像统计与元数据只提供线索，不能据此确认AI生成或真人拍摄。'
    return result
