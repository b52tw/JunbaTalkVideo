> 本封包新增 v0.8.3 第一階段核心審閱內容，請先看 PHASE1_REVIEW.md 與 Verify_Mask.cmd。下文 v0.8.2 是上一版執行程式說明，本輪未編譯或打包 EXE。GitHub 建置亦需手動勾選已確認遮罩才會執行。

# 峻爸 AI 對嘴短影音 v0.8.2 執行修復版

修復 GPU 原生中止（含 0xffffffff）未進入 CPU 回退的缺口：前一工作程序 wait 完全退出後，緩衝 1 秒，再用新 CPU 程序重試；NPU 聯動關閉。OpenVINO NUM_STREAMS=1（CPU／GPU）、支援的編譯執行緒=1、CPU 推論=1～2，ORT=2／1。啟動閘門會等 Go 記憶體預算掛接成功才啟動 SDK 子行程。未降低嘴型輸出解析度。

剩餘時間 Go 永遠回傳 local_eta（含 0），介面缺值／非數字顯示 0 秒，階段切換清除舊進度。異常時可按「下載錯誤報告」，內容只有本專案允許的日誌與設定，沒有素材、語音檔或 API Key；報告仍可能含本機路徑，請自行決定是否分享。

啟動：先關閉舊程式，再開 JunbaTalkVideo_v0.8.2.exe，不須刪除 AppData 或重裝已下載模型。載入舊專案確認預算為「自動 4～6 GiB」後重新生成。如仍失敗，請下載錯誤報告供定位；單憑截圖無法證明是記憶體上限或驅動錯誤。

本輪先交 Source／GitHub Builder，再做語法與單格假資料檢查。雲端不再跑完整大模型或長片。工程規則見 DEVELOPMENT_RULES.md；動作轉移分階段規劃見 MOTION_ROADMAP.md。

新增 Lab 嘴周肌膚均值／標準差限幅對齊、上眼皮摺痕動態暗部漸層（最高 Alpha 0.15），皆可關。這些是視覺補償，不是真實牙齒或閉眼紋理重建。

「呼吸＋語音重音肩動」以 3.6 秒正弦、RMS／基頻與阻尼近似 2D 肩頸微動，位移不超過 5 px；GrabCut 前景、邊緣羽化與已知背景外推保護背景，遮罩不可靠則保持原姿勢。全身揮手／轉身與參考影片骨架驅動尚未完成，沒有將 DWPose 偵測冒稱成動作生成。

開啟時自動列舉已安裝 SDK 的 OpenVINO 裝置與 ORT providers；「硬體加速診斷／Self-Check」只執行小於 1 MB 的 Conv 1×1＋ReLU 微型圖。模型完整性僅大小比對或選用串流 SHA256，不載入真實權重。報告與錯誤存於 AppData/JunbaTalkVideo/diagnostics，可由介面下載 JSON。

轉換、五官／動作係數、渲染、局部修復以子程序串行。SadTalker 關鍵點階段只建 KPDetector／MappingNet，不先建立 Generator；渲染階段讀 mmap 陣列，不匯入 Torch，除非明確需要 PyTorch CPU 備援。轉換失敗改 CPU 路線；DLL／GPU 失敗以退出碼 75 結束原工作程序，再由新 CPU 程序重試，顯示原因及實際 CPU；若 CPU 也不可用會保留錯誤，不假稱成功。

OpenVINO 明確 ENABLE_MMAP=True；IR 與選用修復快取以 FP16 壓縮，CPU 可能仍 FP32 運算，其他暫存與 GPU 記憶體並不會因此全部減半。預設 Windows 子程序族私有提交預算依當下可用 RAM 的一半（整數 GiB）動態選 4～6 GiB；查詢失敗用 4 GiB，自檢仍 200 MiB。可手動選 2.5／3／4／5／6 GiB。舊專案舊版預設 3 GiB 在介面載入時切換自動；本版手動設定保留。這不是實體 RAM、映射頁、共享 GPU 或整機上限。超出預算可能無法生成，需要在本機比較引擎或手動提高預算，不能保證所有模型都在 3 GiB 內。

GitHub 工作流程現在整合 Go GUI＋獨立 Embedded Python／OpenVINO／DirectML，提供完整 EmbeddedPortable 資料夾 ZIP，且先上傳源碼封包。此環境只交付交叉編譯 EXE、小版 ZIP 與完整 Builder；預裝 Windows SDK 的 EmbeddedPortable 由 windows-latest 原生建立。詳見 GITHUB_BUILD.md。

以下保留前版操作歷史；若內容衝突，以本頁上方與 v0.8.2 文件為準。

---

# 峻爸 AI 對嘴短影音 v0.8 重建測試版

作者：峻爸。Windows 10／11 x64。開啟 JunbaTalkVideo.exe，先關閉舊版；既有 MuseTalk 模型、專案、音色沿用。SadTalker 按一次檢查／修復，會沿用大權重、升級三個套件。安裝由程式完成，不需自行裝 Python 或編譯模型。

此包是工作環境重置後重新建立的測試版本。原先曾完成的短片與性能數據不作為這次封包的重新驗證；本次已重跑的檢查見 VALIDATION_v0.8.json。尚未重跑完整神經模型，亦未實測 Windows／Intel GPU／NPU，不能保證無違和或所有電腦都能執行。

v0.8 新增：SadTalker／MuseTalk 只融合嘴部並保留原眼與膚質；可選羽化／泊松與嘴唇／下顎範圍。SadTalker 嘴部模式固定原頭姿，整臉模式會改眼部與膚質。真人 GFPGAN／卡通細節採限幅高頻殘差，預設可關，不能還原真實牙齒。未加入 CodeFormer。

可選 0.10 秒閉／0.25 秒開的貝茲輕眨，休止 3.5～6 秒（固定種子的隨機序列），可聯動眉上區 1～2 像素。仍為 2D 變形，預設眨眼關閉，不能補出照片中不存在的完整閉眼皮膚。自然眨眼與肢體建議使用原本有動作的影片。

SadTalker 可選 OpenVINO、ORT DirectML、PyTorch CPU。首次會由程式自動轉換 ONNX／IR，另約 0.9 GB 快取；3D 變形保留三線性取樣，不是省略動作。五官／語音仍用 CPU。MuseTalk 保留原生 OpenVINO 磁碟映射，新增 DirectML 比較後端與選用 NPU 語音分工。DirectML 僅適用顯卡，NPU 請選 OpenVINO；無額外 API 費。

兩引擎分開安裝 DirectML：選引擎 → 展開 v0.8 選項 → 安裝／修復 → 測試所選後端。小模型完成後回報實際裝置，大模型首格完成後顯示 provider 節點；CPU 分區會明示。小模型成功不代表完整模型一定可跑。Intel 驅動與相容性仍需您的實機驗證。

逐階段程序、mmap_mode='r' 陣列、嘴部 PNG 暫存與 FFmpeg pipe 沿用，避免把長片所有大影格留在 RAM。低干擾預設 2 執行緒。MuseTalk 預設至少 6 GiB 可用 RAM，一般 CPU 兩階段至少 8 GiB；SadTalker 至少 5 GiB。整合 GPU 也使用系統 RAM，這些門檻不保證任何素材都足夠。

保留 Windows 文字語音／錄音／匯入音檔、本機 OpenVoice 音色複製、卡通升降調、字幕燒錄／SRT、MP4／WMV／MOV／MKV／WebM／AVI／M4V／無聲 GIF 與獨立轉檔。字幕錄音模式依講稿估時，不會自動聽寫。不是控制 Windows Media Player；轉檔由內嵌 FFmpeg 完成。一般手機優先 MP4；EXE 只適用 Windows。

建議先用幾秒短句、只融合嘴部、羽化、眨眼／身體微動／細節關閉做比較。原圖保留只指編碼前嘴部遮罩外，影片有損壓縮後像素可能不同；開啟其他動作會修改對應區域。LatentSync 需要 NVIDIA／18 GiB 專用顯存，未提供驗證過的 Windows 一鍵安裝。

GitHub Builder 包含完整原始碼、requirements.txt、PyInstaller spec 和 windows-latest 工作流程；主程式是 Go，原生建置不需 PyInstaller。操作見 GITHUB_BUILD.md。未重跑部分與原有功能操作見以下 v0.7 歷史說明；遇到衝突以前述 v0.8 說明為準。

---

# v0.7 歷史操作說明

Windows 10／11 x64；解壓後開啟 JunbaTalkVideo.exe。原有 MuseTalk 模型、專案、錄音與音色可沿用。先關閉舊版，確認頁面標示 v0.7。介面隨視窗縮放；產出 MP4 可供電腦與手機播放，EXE 本身只適用 Windows。

## 這次修改

- SadTalker 512 可在程式中一鍵安裝：選模式 → 按「一鍵安裝」→ 等待下載／檢查完成。不需要系統 Python、命令列、手填路徑或編譯器。使用獨立環境，CPU 推論，不修改原有 OpenVINO。鎖定 72 個 Windows 套件、官方來源與 SHA256，另外下載原始碼及四個必要權重。總下載約 1.6 GB，預留至少 5 GB 磁碟；安裝中斷可按同一按鈕續傳／修復。安裝鎖由作業系統釋放，不必自行刪除 lock 檔。
- SadTalker 分成五官／3D 係數、語音動作、渲染三個獨立程序；渲染一次只留一格，直接寫入 FFmpeg。這是頭部表情比較模式，只接受照片；不是全身動作模型，也不保證每張素材的唇齒比 MuseTalk 好。預設 CPU 2 執行緒、較低優先序；CPU 渲染非常慢。影片／Intel GPU 或 NPU 加速請使用 MuseTalk，SadTalker 不會使用 Intel GPU／NPU。
- LatentSync 1.6 仍需 NVIDIA CUDA 與至少 18 GiB 專用顯存；Intel Arc／NPU 不能執行。本版沒有可驗證的 Windows 自動安裝，不把此選項假標為已安裝；未配置時停用。這一項尚未完成，沒有要求您自行補裝。
- MuseTalk 預設改用磁碟映射 IR 權重＋逐階段處理。首次額外建立約 1.8 GB 磁碟快取；轉換程序結束後才開始神經推論。既有 ONNX 模型沿用，不重新下載。相同兩格、相同 CPU 2 執行緒測試：原 ONNX 推論程序峰值 6,936.5 MiB，IR 5,403.5 MiB，下降約 22%；輸出 latent 逐值相同。首次快取建立峰值 3,250.4 MiB。這是程序 RAM，不是全部 GPU／NPU 或整台電腦佔用；預設模式仍需至少 6 GiB 可用 RAM，一般兩階段 CPU 需 8 GiB。
- 啟動不載入神經權重，不自動查詢所有歷史專案；預覽不預先解碼影片，輪詢不重設影音來源。「最近的專案」改成按需展開的最近 8 個可開啟專案，保留舊檔案。
- 輸出與獨立轉檔各有格式選單：MP4／WMV／MOV／MKV／WebM／AVI／M4V，另保留無聲 GIF。頁首可直接跳到格式或轉檔。M4V 明確使用 MP4 封裝＋H.264／AAC；AVI 使用 MPEG-4 Part 2（Xvid 標籤）／MP3，方便較舊軟體讀取。轉檔不呼叫付費 AI。WMV 改用高品質可變量化、WebM 從 CRF 32 改為 20，減少再次壓縮的損失；仍不會提高原始神經嘴型資訊。
- 照片輕眨改為約 0.5 秒的非對稱慢動作、約 6～8 秒間隔，減少閉眼幅度，預設關閉；SadTalker 眼部動作也使用同一個低頻驅動。MuseTalk 照片仍是 2D 眼部變形，不能憑空還原閉眼皮膚；要保留真實自然眨眼，優先用已有眨眼的原始影片。
- 提供真人、卡通、快速試做三組 MuseTalk 參數預設，方便固定條件比較。它們是設定組合，沒有換成新的神經模型；卡通細節補強仍需按 9 MB 安裝按鈕，預設不強制使用。

## 建議操作

1. MuseTalk：安裝本機模型與 AI 五官定位，換素材後檢查臉、嘴與眼的定位。卡通可能需手調；影片目前只定位第一格，人物大幅移動會失準。
2. 準備語音／錄音 → 選真人或卡通預設 → 先製作一小段，檢查嘴角、牙齒、鼻子與邊界，再視需要開啟局部細節模型。加大輸出尺寸不會改變原生 256 臉部推論的資訊量。
3. 需要試另一種學習式頭部／表情：選 SadTalker，自動安裝後使用照片與相同語音比較。其眼、頭部、嘴由動作模型產生，可能改變表情或人物細節；512 是渲染尺寸，並非精準保證。SadTalker 自動安裝版可選輸出長邊，先限制大圖尺寸以降低記憶體，原生臉部渲染固定 512／25 FPS。內建五官框、2D 動作、GPU 選單及原圖首尾還原只適用 MuseTalk，在 SadTalker 下停用。
4. 選影片格式、字幕與比例。現有影片可直接在下方轉檔，不需要先製作語音或安裝 AI 模型。
5. 舊專案收合於「需要時開啟已保存的專案」。選成品後，可按重新製作，複用語音與素材做比較。執行期間暫停切換專案，避免誤改工作；無需刪除既有成品。

## 本次驗證範圍

本次以同一張真人首格與 1.2 秒語音，完成 SadTalker 30 格／25 FPS（渲染峰值 3,451.4 MiB，開發機 8 執行緒約 575 秒）與 MuseTalk 15 格／12 FPS 的真實推論。兩者影格率不同，不是精確對嘴評分；短片未涵蓋眨眼或長片穩定性，SadTalker 仍會改動膚質與表情。程式預設 SadTalker 為較慢的 2 執行緒。

實際神經模型測試與程序記憶體、Windows 套件清單、格式／字幕、介面與壓縮檔校驗見 VALIDATION_v0.7.json。Windows x64 EXE 是在 Linux 交叉編譯；本環境沒有 Windows、Intel GPU／NPU 或 NVIDIA 顯卡，不能宣稱已在您的硬體端到端測過。安裝器的依賴均為下載驗證過的 Windows wheel，不需在您的電腦編譯。

## 本機免費聲音複製（v0.3 新增）

本版已加入 OpenVoice V2 神經音色複製，不必使用 Fal、ChatGPT 或 Gemini API。先用 Windows 中文桌面語音把新台詞念出來，再以 OpenVoice 將音色轉換成參考錄音的特徵。這會保留基礎朗讀的語速、咬字、情緒與大致停頓；不是單純升降音調，也不是背景持續訓練。不保證與本人完全相同，請先試聽。

1. 開啟 EXE，展開「錄我的聲音／本機免費聲音複製」。
2. 按「安裝／修復本機聲音複製」。模型約 131 MB；若尚未安裝共用環境，另約 185 MB。已裝 v0.2 的環境和對嘴模型會沿用，無需再下載 1.8 GB 對嘴模型才能單獨產生聲音。
3. 錄製自己的 5～60 秒清楚單人說話，按「以上方錄音／音檔作為樣本」，或直接在「新增音色的錄音樣本」選檔。請使用無配樂、少回音、收音音量穩定的內容。
4. 輸入音色名稱和新的台詞。基礎語音選「讓 Windows 朗讀上方新台詞」，使用原有 Windows 聲音／語速選項；沒有中文桌面語音時，可改選匯入已念好台詞的音檔並更換音色。
5. 運算裝置使用「自動」。按「用我的音色朗讀新台詞（本機免費）」。模型、文字、原始錄音與音色特徵都在本機處理，不會上傳。
6. 完成後試聽、下載 WAV；或按「使用這段語音製作影片」，回到照片／影片流程製作對嘴。音色已保存在本機選單，下次選該名稱並換新台詞即可沿用。

只有聲音複製時，啟動檢查至少 2 GB 可用記憶體；MuseTalk 預設磁碟映射對嘴需 6 GiB，一般兩階段 CPU 需 8 GiB。測試環境約 4.9 秒音訊的聲音複製程序峰值記憶體約 571 MiB，僅代表該 CPU 測試，實際 GPU／NPU 編譯需求可能不同。聲音與對嘴在同一程式內依序執行；請勿同時開多份 EXE。

聲音運算使用 OpenVINO。自動模式優先嘗試 Intel GPU；小型參考編碼器若 NPU 支援，可由 NPU 分工。固定 256 格模型會先查詢 NPU 算子支援、編譯並實際推論，再顯示使用的裝置；不支援時在自動模式嘗試其他裝置，最後才用 CPU。明確選 GPU／NPU 時，失敗會直接顯示錯誤，不會暗中換裝置。畫面顯示各模型實際裝置與退回原因。

若只看到 CPU，可先更新 Intel 官方顯示／NPU 驅動並重新開啟程式：https://www.intel.com/content/www/us/en/support/detect.html 。偵測到 NPU 不代表全部聲音模型都相容。您的新版截圖顯示 Intel Arc 130T GPU 已實際執行 UNet、VAE 與 Whisper；此開發環境仍只有 CPU，NPU 相容性需在您的電腦驗證。

## 製作對嘴影片

1. 解壓縮，開啟 `JunbaTalkVideo.exe`。
2. 按「首次安裝／修復本機模型」，保持連網直到完成。EXE 內含獨立 Python，不會更動系統 Python；首次另外下載約 185 MB 套件及 1.83 GB 模型，建議預留至少 5 GB 磁碟空間。
3. 按「偵測可用裝置」。選「自動」會優先使用 OpenVINO 可用的 Intel GPU；模型編譯失敗時才退回 CPU。實際裝置會逐模型顯示。選定 GPU 或 NPU 時，失敗會直接顯示原因，不會冒稱成功使用該裝置。
4. 使用內建卡通圖片，或換照片／影片。按「顯示臉部框選」，拖曳框出包含眼睛、嘴巴和下巴的完整臉部。內建圖片已有預設框。
5. 輸入 1～2 句台詞，使用 Windows 中文桌面語音，或錄音／匯入聲音。按「先製作語音、確認素材」，試聽語音。
6. 精緻輸出建議 25 FPS、長邊 1920；只試流程可選 6 FPS、512。選字幕、格式和比例，再按「使用本機硬體生成影片」。完成後下載成品或跨裝置播放包。

模型安裝完成後，本機語音、錄音、音色複製、變聲、對嘴、字幕和轉檔可離線執行。它們不需要 ChatGPT、Gemini 或其他雲端 API Key，也不收每次生成的 API 費用。首次安裝下載的是執行套件與模型，不會上傳您的素材。

## 您的 Intel 硬體

您最新截圖顯示 Intel Core Ultra 5 225H、Intel Arc 130T GPU、Intel AI Boost NPU，約 15.4 GB 可用總記憶體；截圖當時約剩 5.4 GB。開始生成前請先關閉其他程式。本機 GPU／自動模式檢查至少 6 GB 可用記憶體，預設磁碟映射 CPU 也是 6 GiB，一般兩階段 CPU 模式至少 8 GiB；這是啟動門檻，不保證每種裝置都不會遇到記憶體不足。整合式 GPU 也會使用系統記憶體。

- Intel GPU：由 OpenVINO 執行 MuseTalk、VAE 和語音特徵模型。能否使用取決於 GPU 型號與驅動。小型位置編碼圖固定使用 CPU，會如實列出。
- Intel NPU：只有 OpenVINO 偵測到才出現在選單，屬實驗相容性選項。必須成功編譯所需模型才能完成工作；不保證目前所有算子都受支援。對嘴的自動模式不會使用 NPU；聲音複製自動模式可在相容時嘗試 NPU 參考編碼。
- Intel Quick Sync：MP4／MOV／MKV 的輸出轉檔會先做實際編碼測試；可用才啟用。若轉檔失敗會重試 CPU，並顯示退回情形。字幕、影像合成、其他格式等仍可能使用 CPU。
- 畫質與速度：臉部模型工作區域為 256×256；選 720／1080 輸出不會補出真實新細節。時間依電腦而異，畫面會顯示進度及按已完成格數估算的剩餘時間，首次編譯不包含在預估內。

已以真實模型在 Linux CPU 上生成約 4.9 秒有聲影片；這不是 Intel GPU／NPU 效能測試。尚未在您的 Windows 電腦上實測。若看不到 GPU／NPU，先更新 Intel 官方顯示／NPU 驅動，再按偵測；不要把「偵測到」當作「所有模型都能執行」。

## 聲音與字幕

- Windows 內建中文桌面語音；若沒有中文聲音，可錄音或匯入音檔。
- 接受 WAV／MP3／M4A／AAC／OGG／FLAC／WebM；3～60 秒、15 MB 以內。
- 原聲、可愛卡通、活潑尖聲、低沉角色。音調效果不是複製音色。
- 可錄麥克風並下載 WAV；已匯入的 Siri 風格或其他 AI 語音也能使用。未內附 Apple Siri 專有聲音。
- 字幕可燒入影片並另存 SRT。Windows 語音優先依進度事件對齊；錄音、雲端合成或無事件資料時依講稿估時，請試看確認。沒有填講稿不會自動聽寫。
- 「用我的音色朗讀新台詞」現在可使用免費本機 OpenVoice V2；另保留收折的 Fal 雲端模式供選用，只有該模式需要付費。

## 素材與輸出

照片支援 JPG／PNG／WebP；影片支援 MP4／MOV／WebM，素材限 20 MB、講稿限 800 字。本機模型使用固定臉部框；影片中人物大幅移動、側臉、遮嘴、多人物時會失準，請先裁切穩定的單人近景。片段比聲音短時循環原影片。卡通或真人仍可能有嘴型不自然、局部柔化或閃爍，請用短句試做並調整嘴框。

可選 MP4、WMV、MOV、MKV、WebM、AVI、M4V、GIF；GIF 沒有聲音。可保持比例或加黑邊輸出 9:16、16:9、1:1。WMV 可供支援該格式的 Windows 播放器播放；程式用 FFmpeg 轉檔，不是自動控制 Windows Media Player。

單獨轉檔限 500 MB，不需要安裝對嘴模型。輸出的標準 H.264／AAC MP4 可在一般電腦和手機播放器觀看；響應式播放包含 `player.html` 及影片。手機不支援開本機 HTML 時，直接開 MP4。同一 EXE 不能安裝到 Android、iOS、macOS。

## 儲存與中斷

專案位於 `%APPDATA%\JunbaTalkVideo\output\專案ID`；「開啟專案資料夾」可查看素材副本、講稿、WAV、SRT、原始 `video.mp4`、成品 `export.格式`、`local-config.json` 和 `local.log`。

本機環境及模型沿用 `%APPDATA%\JunbaTalkVideo\local-v0.2`。本機音色及原始樣本保存在 `%APPDATA%\JunbaTalkVideo\voice-profiles\音色ID`，含 `sample.wav`、`embedding.npy`、`profile.json`；請保留錄音原檔以便另機重建。安裝失敗請看 `install.log` 並重試；模型下載支援續傳且以固定 SHA256 驗證。上次安裝若異常結束而留下 `install.lock`，確認所有程式視窗已結束後，刪除該鎖定檔再重試。如果 OpenVINO 匯入失敗，檢查 Intel 驅動與 Microsoft Visual C++ x64 執行階段：https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist 。

v0.4 的離線工作程式明確停用 OpenVINO 的可選遙測匯入；本機推論不需要對外連線。

本機生成可按「停止本機生成」；重新開啟時會把中斷專案恢復為可生成狀態，不會付費重送。原始素材不會被覆寫。同一專案重轉相同格式會更新該 `export` 檔，如需多版請先下載另存。

雲端模型仍保留在「選用：雲端模型」區域，只有明確按付費按鈕才會送出。金鑰僅在本次執行期間使用，不寫入專案。雲端中斷可查詢原任務 ID；結果不明時先到供應商控制台確認，避免重複計費。

## 驗證與限制

v0.4 使用真實 MuseTalk 與 Real-ESRGAN 完成 3 秒語音、75 格、25 FPS 的 CPU 端到端測試，先退出對嘴程序，再執行細節合成與字幕轉檔；成品 1286×1920、4.12 秒（含片頭片尾）。第一格與最後一格對縮放原圖的 PSNR 約 39.76 dB，開始 0.35 秒為靜音，字幕由 0.56 秒開始。另以合成影像檢查鼻子／眼睛／背景不被嘴部遮罩改動，以及可選身體微動與 UI 動作選單。這些檢查不代表所有嘴型都已達到無違和。

已完成真實 OpenVoice CPU 音色複製、樣本保存與沿用、WAV 接到影片準備流程，及真實 MuseTalk CPU 生成、H.264／AAC 輸出、Go 整合測試；另測試六種格式、字幕燒錄、音訊保留、音調效果、工作階段限制、中斷恢復、模擬雲端流程及 Go race detector。UI 經 JavaScript 語法與 DOM 流程檢查。Windows EXE 已交叉編譯，未做數位簽章；Windows 可能顯示發行者不明。

音色轉換測試使用合成英文基礎音與合成參考音，未取得您的真實錄音，不能判定與您的相似程度。開發環境未在真實 Windows 上驗證內附 Python 的首次安裝、中文 SAPI、NPU／Quick Sync 或麥克風；Intel GPU 執行已有使用者截圖佐證，仍不代表每種 GPU 都相容。未完成真實瀏覽器的視覺渲染驗證，也未進行付費雲端端到端實測。請先用短句測試，勿把測試版視為所有電腦皆可直接成功的正式產品。

## 原始碼與編譯

`source` 內含 Go 原始碼、介面、本機 Python 程式、模型固定版本與 SHA256。`build.ps1` 以 Go 1.24 以上建立單一 EXE。為控制 ZIP 大小，源碼內省略已嵌入 EXE 的 FFmpeg 與字型壓縮包，編譯腳本會依固定 SHA256 重新取得；獨立 Python 與 pip 檔案有附上。

開發測試：先取得編譯腳本所需的內嵌 FFmpeg／字型檔，再執行 `go test -race -v ./...`。實際模型整合測試需另外設定 `JUNBA_TEST_LOCAL_MODELS`、`JUNBA_TEST_LOCAL_PYTHON`、`JUNBA_TEST_LOCAL_AUDIO`，並在 Linux 執行。本機聲音整合測試另需 `JUNBA_TEST_VOICE_MODELS` 與 `JUNBA_TEST_VOICE_REFERENCE`。第三方資訊與授權見 `SOURCES.md` 和 `third_party`。


開發檢查：`python tests/test_face_geometry.py`、`python tests/test_isolated_workers.py`；`tests/test_ui.cjs` 使用 jsdom，可在獨立目錄安裝後透過 NODE_PATH 執行。真實五官 API 測試設定 `JUNBA_TEST_FACE_MODELS`（含 landmarks98.xml/bin）、`JUNBA_TEST_FACE_SOURCE` 和 `JUNBA_TEST_LOCAL_PYTHON`。CLI 相容測試使用替身；新版 SadTalker 另做真實神經模型驗證，見 VALIDATION_v0.7.json。
