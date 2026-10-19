# 官方資料與第三方元件

模型 API（本版依以下官方文件串接；尚未執行付費實測）：

- https://fal.ai/models/fal-ai/kling-video/ai-avatar/v2/standard/api
- https://fal.ai/models/fal-ai/sync-lipsync/v2/api
- https://fal.ai/models/fal-ai/minimax/voice-clone/api
- https://fal.ai/models/fal-ai/minimax/speech-02-hd/api
- https://fal.ai/docs/documentation/model-apis/inference/queue
- https://learn.microsoft.com/en-us/dotnet/api/system.speech.synthesis.speakprogresseventargs
- https://support.microsoft.com/zh-tw/windows/apps/windowsmediaplayer/file-types-supported-by-windows-media-player

FFmpeg 9.0.2 Essentials Windows x64（獨立子程序；GPL v3）：

- 官方下載頁：https://ffmpeg.org/download.html
- 散布者：https://www.gyan.dev/ffmpeg/builds/
- 對應下載：https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip
- 下載 ZIP SHA256：60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba
- 對應 FFmpeg 原始碼：https://github.com/FFmpeg/FFmpeg/commit/946fcce07b
- 可下載該原始碼壓縮包：https://github.com/FFmpeg/FFmpeg/archive/946fcce07b.tar.gz
- 上游建置說明與所有 configure 參數：third_party/README.txt；授權：third_party/LICENSE。

Noto Sans TC（SIL Open Font License 1.1）：

- 上游原始碼及字型：https://github.com/notofonts/noto-cjk
- 字型：https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/Variable/TTF/Subset/NotoSansTC-VF.ttf
- 原始 TTF SHA256：ac091cc8cd19e848202afc8fe6d3809b4526c8fdbdb4be82da20c4f785949591
- 授權：third_party/Noto-OFL.txt。

使用者提供的卡通照片僅作本使用者程式的預設素材，未授予第三方圖片權利。程式自有原始碼適用 MIT，第三方元件依各自授權。

## v0.2 本機對嘴

- MuseTalk 官方：https://github.com/TMElyralab/MuseTalk 。官方程式為 MIT，訓練權重依官方說明可使用；其他相依元件仍依各自條款。
- 本版使用的 ONNX 轉換為 DgDev91 社群版本，並非官方 Windows 安裝包：https://huggingface.co/DgDev91/MuseTalk-ONNX/tree/79550fb510b18ea14e425a1ddf8d99693fb5cdc8 。五個 FP16 模型的檔名、固定來源、大小及 SHA256 完整列於 `local/models.json`。模型由使用者首次安裝時直接下載，未包含在發行 ZIP。
- OpenVINO 2026.4.1：https://docs.openvino.ai/ ；Windows cp311 wheel 與全部執行相依套件已核對可下載。安裝版本及 Windows wheel SHA256 見 `local/requirements.txt`。
- NumPy 1.26.4、SciPy 1.14.1、OpenCV headless 4.10.0.84；由 PyPI 安裝原套件及所附授權。未使用 Wav2Lip 原始非商用模型。
- Whisper MIT 及 mel 濾波器：https://github.com/openai/whisper/tree/v20240930 。語音特徵由模型運算；未以此做自動聽寫字幕。
- Python 3.11.9 Windows embedded x64：https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip 。SHA256：009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b。PSF 授權包含於原 ZIP 與 `third_party/Python-LICENSE.txt`。
- pip 25.0.1：https://pypi.org/project/pip/25.0.1/ 。獨立環境中解壓 wheel 啟動，不使用系統 pip。pip MIT 授權見 `third_party/pip-LICENSE.txt`。
- Windows 工作管理員截圖用於判斷使用者硬體，不納入軟體發行包。

## v0.3 本機免費聲音複製

- 官方 OpenVoice V2：https://github.com/myshell-ai/OpenVoice 。官方 V1/V2 為 MIT，可免費商用；此處使用音色轉換模型，基礎文字朗讀仍由 Windows 本機語音負責。
- 上游參考流程：https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvoice2-and-melotts 。本程式未散布該完整 notebook 或 MeloTTS，而是以 NumPy/SciPy 實作頻譜前處理，再由 OpenVINO 執行固定形狀的 ONNX 模型。
- ONNX 轉換來源：https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2/tree/34d010c192c97f763207f488f6057fd07fee42ad 。使用 FP32 `tone_ref_encoder.onnx`（3,259,275 bytes）與 `tone_converter.onnx`（128,051,288 bytes）；固定下載版本、SHA256 與大小見 `local/voice_models.json`。未將社群模型卡的語者相似度數字當作對使用者本人音色的保證。
- MIT 授權見 `third_party/OpenVoice-LICENSE.txt`。模型在首次安裝時下載，本 EXE 不內嵌權重，且執行不使用 PyTorch、CUDA 或付費 API。

## v0.4 選用卡通細節補強

- Real-ESRGAN 官方：https://github.com/xinntao/Real-ESRGAN ，BSD 3-Clause，授權見 `third_party/RealESRGAN-LICENSE.txt`。
- 使用 `RealESRGAN_x4plus_anime_6B` 的社群 FP16 ONNX：https://huggingface.co/skillsafe-ai/realesrgan-x4plus-anime-6b/tree/c001be50fae1c7a2bae8b2b4dec5e6fcbdbcca96 。固定大小 9,031,567 bytes；SHA256 `1777c7dffecdf356c234e3387d456a196ed9b72f9f14ed96bc097b9798b7ddbf`。完整下載設定見 `local/quality_models.json`。權重在選用安裝時下載，不包含於 EXE。
- 眨眼、視線微移與點頭為本程式的可選 2D OpenCV 變形，不使用 LivePortrait、InsightFace、動作生成或人臉偵測權重。

## v0.4.1 Windows 模組路徑修正

- Python 官方關於 `._pth` 隔離與模組搜尋路徑的說明：https://docs.python.org/3.11/library/sys_path_init.html#pth-files 。本版直接從工作程式的 `__file__` 取得模組目錄，不依賴目前工作目錄、系統 Python 或 `PYTHONPATH`。

## v0.5 記憶體分段與真人紋理參考

- GFPGAN 官方：https://github.com/TencentARC/GFPGAN 。Apache 2.0 及第三方說明見 `third_party/GFPGAN-LICENSE.txt`（官方 v1.3.8 的 LICENSE，涵蓋本使用的 v1.4 模型系列）。
- 使用 FaceFusion 社群提供的 GFPGAN 1.4 ONNX：https://huggingface.co/facefusion/models-3.0.0/resolve/728b9659bd9691bf32cbf7f61af478d94b7ba81e/gfpgan_1.4.onnx 。固定大小 340,299,087 bytes；SHA256 `accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385`；完整設定見 `local/real_models.json`。權重不包含於 EXE，由使用者選用安裝時下載。
- GFPGAN 的 512×512 輸入為 RGB、[-1,1]；本程式經 OpenVINO 執行，只取修復結果受限的高頻殘差作嘴部紋理參考。沒有使用 FaceFusion 的換臉、偵測器或身分替換流程。修復模型不保證重建真實細節或保留每一格開口。
- Python 程序以磁碟陣列交接，Windows CPU 優先序使用 SetPriorityClass；程序峰值使用 GetProcessMemoryInfo 的 PeakWorkingSetSize。非 Windows 測試以 resource.ru_maxrss 量測。這些數據不等同全部 GPU／NPU 記憶體。

## v0.6 AI 五官定位與其他模型／免費入口

- OpenCV YuNet 官方：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet 。以 `FaceDetectorYN` 做空間偵測，並非身分比對。內嵌 232,589 bytes ONNX 固定來源：https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx 。SHA256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`，MIT 授權見 `third_party/YuNet-LICENSE.txt`。
- Intel Open Model Zoo facial-landmarks-98-detection-0001：https://docs.openvino.ai/2023.3/omz_models_model_facial_landmarks_98_detection_0001.html 。BGR 64×64 輸入、98×16×16 熱圖輸出。FP16 XML／BIN 依官方 model.yml 的 SHA384 核對，再產生本程式 SHA256；完整來源與 SHA256 見 `local/face_models.json`。下載約 20.8 MB，權重不內嵌；Apache 2.0 見 `third_party/Intel-OMZ-LICENSE.txt`。
- 熱圖座標解碼參考官方 C++ 範例：https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/gaze_estimation_demo/cpp/src/landmarks_estimator.cpp 。使用峰值與局部梯度修正；區域框仍為本程式加邊界的估計範圍，不等於逐像素分割。
- SadTalker 官方與 CLI：https://github.com/OpenTalker/SadTalker 、https://github.com/OpenTalker/SadTalker/blob/main/inference.py 。未散布其專案程式或權重，只以參數呼叫使用者自行安裝的環境。GPU 路線使用 CUDA，本接入可指定 CPU。
- LatentSync 1.6 官方：https://github.com/bytedance/LatentSync 。官方說明原生 512、推論最低 18 GB VRAM；本程式保守檢查 18 GiB，需 CUDA。未散布其程式或權重，只提供外部 CLI 接入。
- HeyGen：https://www.heygen.com/pricing 。2026-10-18 官方列 Free 每月 3 支、每支最長 1 分鐘；進階功能有試用限制，API 方案另計。
- LivePortrait：https://github.com/KlingAIResearch/LivePortrait 、https://huggingface.co/spaces/KlingTeam/LivePortrait 。查核當日官方展示 Running on Zero。共享免費額度／排隊說明：https://huggingface.co/docs/hub/spaces-zerogpu 。此處提供網站連結，不在本程式內執行該模型。
- D-ID：https://www.d-id.com/pricing/studio/ 。2026-10-18 查核官方試用入口及試用浮水印說明；不是永久無限免費。


## v0.7 自動安裝與磁碟映射

- SadTalker 固定官方 commit cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8；原始碼 ZIP 與必要模型的來源、大小、SHA256：local/sadtalker_models.json。安裝時下載，非內嵌整個神經模型。Apache 2.0 授權見 third_party/SadTalker-LICENSE.txt。執行包裝器新增分程序與逐格回貼，未修改上游模型權重。
- 72 個 Windows x64／通用 Python wheel 固定於 local/sadtalker_wheels.json，逐一實際下載並核對。來源 PyPI files.pythonhosted.org 與 PyTorch 官方 CPU wheel。basicsr 1.4.2、filterpy 1.4.5 從官方 PyPI source distribution 建立純 Python wheel，附於 local，使用者不需編譯工具；授權包含在 wheel metadata。
- OpenVINO IR 序列化使用 openvino.save_model，讀取 XML／BIN，讓權重可由磁碟映射。FP16 ONNX 權重原值沿用；本次比較 latent 輸出逐值相同。不是量化或增加模型解析度。
- 本版自動安裝取代 v0.6 要求手動填路徑的流程；LatentSync 自動安裝仍未完成，未宣稱可在 Intel 上運作。

- 缺少 C++ DLL 時才下載並啟動 Microsoft 官方 VC++ Redistributable，使用正常 Windows 授權／安裝介面；固定來源與 SHA256 見 local/sad_runtime.json。


## v0.8 重建

- ONNX Runtime DirectML：https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html ，固定 1.24.4；來源與 SHA256 見 local/directml_wheels.json。序列執行、關閉 memory pattern，profiling 回報實際 provider。
- OpenVINO EP：https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html 。本版保留原生 OpenVINO IR 路線，另選 ORT DirectML，不使用舊內建 OV EP。
- OpenCV seamlessClone：https://docs.opencv.org/4.x/df/da0/group__photo__clone.html 。融合後再由嘴部羽化遮罩限制更新範圍。
- GitHub Actions：https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions ，官方 actions 固定 commit SHA，contents:read。個別檔案 artifact：https://github.com/actions/upload-artifact#upload-an-individual-file-unzipped 。
- PyInstaller：https://pyinstaller.org/en/stable/usage.html ，只作 Windows 可選包裝器，主程式為 Go CGO=0。
- 75 個 SadTalker wheel 在 v0.7 的 72 個上新增 ONNX 1.16.2、OpenVINO 2026.4.1 與 telemetry 2025.2.0。開發測試包不宣稱全部硬體相容。


## v0.8.1 工程與記憶體

- OpenVINO ENABLE_MMAP 官方定義：https://github.com/openvinotoolkit/openvino/blob/master/src/inference/include/openvino/runtime/properties.hpp 。以路徑讀 IR；FP16 是權重壓縮，不是全程 RAM 保證。
- Windows Job Object 提交量限制：https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information 。JOB_OBJECT_LIMIT_JOB_MEMORY 限制程序族可提交虛擬記憶體，不等同 RSS／GPU／映射頁。
- 微型圖為自行產生 ONNX Conv＋ReLU，只有 9 個浮點權重，不使用真實模型。
- 前景保護使用 OpenCV GrabCut 傳統分割、SciPy 距離外推；未整合 AnimateAnyone 或 DWPose 的動作生成，後續規劃見 MOTION_ROADMAP.md。
