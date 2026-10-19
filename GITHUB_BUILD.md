# v0.8.2：Go＋Embedded Python 混合原生建置

1. 解壓 GitHub_Builder.zip，把內容上傳至 repository 根目錄。根目錄須有 go.mod、build.ps1，以及隱藏資料夾 .github/workflows/build_exe.yml，勿多包一層。
2. Actions 允許執行後，push 自動觸發，或選 Build Windows Go + Embedded Python → Run workflow。
3. 流程一開始先產出並上傳 Source-and-GitHub-Builder，即使後續建置失敗仍可先下載源碼。
4. Windows 原生 go build 產出 GUI EXE；另一個獨立 Embedded Python 3.11.9 子模組裝入固定 OpenVINO 2026.4.1 與 ORT DirectML 1.24.4，只做微型 Conv＋ReLU 與一格假資料合成。沒有真實大模型或長片推論。
5. 下載 JunbaTalkVideo_v0.8.2_EmbeddedPortable.zip，解壓同一資料夾後開 JunbaTalkVideo.exe。runtime/base/python 是獨立推論環境，不需系統 Python。新電腦可先做 Self-Check，再按一鍵下載模型；如果已有 AppData 環境則沿用它，避免更動舊環境。

Artifacts 另提供小版 Windows64.zip 與單一 EXE；這兩種沒有預裝完整 SDK，會由程式一鍵安裝。免安裝 EmbeddedPortable 已含 SDK，但不含數 GB 模型權重。SadTalker 仍是獨立的一鍵安裝環境，不把全部 Torch 模型塞入輕量啟動程式。

requirements.txt／JunbaTalkVideo.spec 是可選 PyInstaller 啟動包裝器，Run workflow 勾 pyinstaller_launcher 才額外建立。預設原生 Go EXE 本身已是單一檔，不需 PyInstaller；包裝器不提高畫質，也不替代 runtime 子模組。

Windows-validation-and-small-portable 含微型圖、單格假資料、自檢與 SHA256 報告。GitHub 沒有您的 Intel GPU／NPU，此流程尚未在您的 repository 代跑；不能保證所有電腦 100% 相容。GPU／NPU、完整模型與長片測試只在您的 Windows 本機進行。

腳本從固定 SHA256 來源取得 FFmpeg／字型，SDK 相依套件依固定 Windows wheel SHA256 取得。不需要 AI API Key。GitHub Actions 的額度依您的方案而定。EXE 未簽章，Windows 可能顯示信任提示。
