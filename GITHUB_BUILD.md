# v0.9.0 Windows EXE 建置流程

## 建置產物

根目錄 `VERSION` 是唯一版本來源。Windows 建置成功後，Actions 會上傳：

- `JunbaTalkVideo_v0.9.0.exe`：Windows x64 原生 GUI。
- `JunbaTalkVideo_v0.9.0_Windows64.zip`：小型便攜包。
- `JunbaTalkVideo_v0.9.0_EmbeddedPortable.zip`：獨立 Python／OpenVINO／DirectML 執行環境；不包含數 GB 的 AI 權重。
- `Source-and-GitHub-Builder`：在測試前先建立並上傳的原始碼／建置封包。
- `Windows-validation-and-small-portable`：自我檢查、合成測試報告與 SHA256。

## 流程

1. GitHub Actions 先讀取 `VERSION`，產生 Source／Builder checkpoint，並在測試前上傳。
2. Windows Runner 使用 `build.ps1` 編譯 Go 原生 GUI，執行錯誤回復回歸測試及 EXE headless self-check。
3. `ci/build_runtime.py` 建立獨立 Embedded Python 3.11.9 與固定版 OpenVINO／ORT DirectML SDK；只列舉裝置並跑微型合成圖，不讀取真實神經權重。
4. 合成資料測試驗證背景像素保護、遮罩失敗時安全略過、以及啟動預檢的錯誤訊息。
5. `ci/package.py` 依 `VERSION` 封裝 EXE 與 ZIP。大模型權重仍由使用者依需求另行安裝。

## 首次使用

- 先開啟 EXE，按「安裝本機輕量環境（不下載 AI 權重）」可準備卡通原圖 Puppet 所需的本機套件。
- MuseTalk 需要另外按「安裝／修復完整 AI 模型」下載權重；預設先用 720p／12 FPS 試做。
- SadTalker 使用獨立安裝流程。MuseTalk 主要做對嘴，並不生成任意全身姿勢；較明顯的頭部／表情動作請使用 SadTalker 或雲端 AI。

## 驗證限制

PR 建置驗證編譯、SDK 套件、合成遮罩與啟動預檢，不執行真實 MuseTalk／SadTalker 全模型推論、長影片或 GPU／NPU 壓力測試。完整模型、Intel GPU／NPU、四支素材回歸與長片穩定性仍須在目標 Windows 電腦實測。未測項目不得標示為已通過。

EXE 未簽章，Windows 可能顯示信任提示。建置下載的權重與雲端 AI 素材處理，請依各引擎的授權與隱私條款確認。
