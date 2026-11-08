# 离线图像证据检查

保留EXIF/C2PA存在性、FFT、ELA、噪声与启发式特征、证据融合和JSON输出；移除神经模型自动下载。

## 使用

Python 3.11+。在本目录建立独立虚拟环境：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python cli.py example.json
.venv/bin/python -m unittest discover -v
```

`example.json`从零编写，仅含虚构输入。可以显式提供自己的同结构JSON文件。输出默认为终端；`--output result.json`使用独占创建，避免覆盖已有结果。CLI失败退出码2。

## 数据与边界

不发现本机账号、不读取环境中的模型密钥、不发送消息、不请求远端数据。只处理调用方明确传入的文件、记录或参数。合成示例不代表任何实际业务结果。不含训练数据、模型权重、照片或GUI。证据模块失败会显示error，不代表检查通过；未独立验证C2PA签名真实性。

## 发布状态

本项目源码采用 MIT 许可证，见 LICENSE。依赖按各自上游许可使用，安装的依赖/模型/字体/媒体不随此源码目录分发。

C2PA的manifest/issuer只作为INFO中性元数据；不授予verified=true，不因相机或AI平台名称判断画面真实/生成。签名错误、未知与空状态均不参与来源评分。

可选C2PA SDK只在具备显式Context时读取，强制关闭远程manifest和OCSP；旧版仅扫描字节标记。解析到声明也仅是中性元数据，不授予签名有效或画面真实结论。[SDK离线设置依据](https://github.com/contentauth/c2pa-python/blob/main/docs/context-settings.md)。

## 2026-11-08 协作修订

- 计数直方图先归一化为概率再计算熵。
- Exif子IFD拍摄字段正确读取。
- 零信号FFT保持中性。
- 小图ELA及超窄图ELA/PRNU使用有效尺寸；结果仍仅为技术线索。

集合根目录 `verify_release.py --project personal-image-evidence` 可复跑合成测试和首次使用示例；本轮以集合验证报告为准。
