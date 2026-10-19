# v0.8 重建測試版引擎

MuseTalk：照片／影片嘴型、原生 256 臉部，OpenVINO 或 DirectML。影片固定第一格框，人物大幅移動會失準。

SadTalker：一鍵安裝／修復，75 個固定 Windows wheel 與四個必要權重，新增 ONNX／OpenVINO；約 1.7 GB 下載，預留至少 6 GB 磁碟含加速快取。沿用 engines/sad512-cpu-staged-v07 目錄，升級 installed.txt 標記。只接受照片、512／25 FPS。嘴部模式保留原眼與膚質；整臉模式用模型表情。轉換由程式自動完成，不需自己安裝外部引擎。

DirectML 是各引擎選用一鍵安裝；不支援 NPU，實際節點分區首格後回報。OpenVINO NPU 屬實驗路線。PyTorch CPU 是 SadTalker 備援，較慢。

GFPGAN／卡通細節需另按模型安裝按鈕，只加入限幅高頻資訊，不保證真實牙齒。泊松可能減少接縫，也可能降低對比；羽化為預設。2D 眨眼仍有變形限制。

本包已重建，完整大模型未重跑，未在 Windows／Intel GPU／NPU 實測。請用短句比較，失敗可選 PyTorch CPU 或原有 MuseTalk OpenVINO，查看專案 local.log。GitHub 原生流程見 GITHUB_BUILD.md。
