# JunbaTalkVideo 工程規則（自 v0.8.1 起）

1. 每批程式與建置腳本修改完成，先執行 `python ci/checkpoint.py --out ../v08-checkpoints/批次名稱`，立即交付並保存 Source.zip、GitHub_Builder.zip，再進行測試；不得只留暫存工作區。
2. 雲端只做語法檢查、編譯與一格合成假資料 Smoke Test。不得下載真實神經模型來跑完整推論，不跑長片或 GPU／NPU 壓力測試。
3. GitHub Windows 流程先上傳源碼封包，再編譯 Go，建置獨立 Embedded Python SDK（OpenVINO＋ORT DirectML），以微型圖測 DLL，最後打包 EmbeddedPortable 資料夾。完整模型權重按需下載，不含於 SDK。
4. 自檢只可列舉 provider／裝置、執行動態 Conv 1×1＋ReLU 微型圖，以及大小／串流 SHA256 核對。不得 open/read_model 真實模型、不得使用 onnx.load 或對大型模型做 checker；路徑模式也不假定永遠無張量記憶體配置。
5. 轉換、特徵提取、渲染、修復以短命子程序嚴格串行；上一程序 wait／退出後才啟動下一階段。Stage B 不匯入 Torch／Audio2Coeff／KPDetector，讀 mmap 陣列與 OpenVINO IR。
6. OpenVINO 明確 ENABLE_MMAP=True；IR 權重預設壓縮 FP16，CPU 計算可能仍 FP32。不得宣稱整體 RAM 一定減半或嘴型不變。
7. Windows Job Object 提供程序族私有提交預算，預設動態 4～6 GiB（v0.8.2 依使用者最新要求變更），診斷 200 MiB。它不是 RSS、GPU、映射頁或整機 RAM 上限。失敗應顯示日誌、保留素材，無法在預算內運行的模型不得冒稱成功。
8. 原有大模型與 Intel GPU／NPU 效能測試由本地 Windows 實機執行，記錄環境、模型、峰值與成品，未測部分必須明示。

補充：加速後端失敗以退出碼 75 交還純標準函式庫 supervisor；必須 wait 前一工作程序結束後，才用新 CPU 程序重試。最多三次，避免無限循環。局部修復失敗也採此方式；CPU 亦不可用時保留錯誤，不冒稱已完成。

原生工作程序非預期退出也須處理：wait 後緩衝 1 秒才 CPU 重試，最多四次總嘗試。NUM_STREAMS=1（CPU／GPU）、編譯執行緒=1（支援時），CPU 推論 1～2。local_eta 永遠有數值，失敗清除過期進度。
