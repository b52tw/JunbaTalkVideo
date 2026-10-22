# v0.9.0 後續實作補充

本文件原本記錄 v0.8.3 的第一階段靜態遮罩審閱。v0.9.0 已將保守前景遮罩接入本機頭部微動，並新增固定背景合成、低幅身體微動、啟動預檢與 Windows EXE CI 回歸測試。

- 遮罩／GrabCut 無法通過可靠性檢查時，會跳過頭部動作，不以整個矩形區域硬做變形。
- 頭部角度限制約 ±1.5°、位移約 2 px；身體微動位移限制約 2 px。
- GitHub Actions 的合成遮罩測試與 EXE 自我檢查，仍不等於真人／卡通真實素材、MuseTalk／SadTalker 完整模型、Intel GPU／NPU 或長片驗收。
- 本機啟動預檢只檢查必要素材、套件、後端、權重檔與磁碟空間，不會載入真實神經模型權重。

此補充不追溯改寫下方 v0.8.3 的歷史審閱結論。請以 v0.9.0 Pull Request 的 Actions 結果及 `VALIDATION_v0.9.0.json` 判定自動化檢查狀態。

---

# v0.8.3 第一階段：唇線核心原始碼審閱包

本包尚未編譯 EXE，不代表已完成一鍵智慧模式。請先檢查靜態遮罩，再由 Windows 本機跑 1 秒模型測試；本輪不跑完整模型或長片。

## 修改位置與函式

| 檔案 | 函式／位置 | 行為 |
|---|---|---|
| local/semantic_mouth.py | heatmap_reliability / from_landmarks | 檢查熱圖集中程度與唇線幾何；品質分數是啟發式，並非校準機率。 |
| local/semantic_mouth.py | _edge_contour / edge_face_anchors | 98 點不足時搜尋影像輪廓。五個偵測點只作搜尋種子；不能直接回傳粗嘴框。找不到可靠輪廓就報錯。 |
| local/semantic_mouth.py | geometry / resample_closed | 輸出外唇、可見內嘴、嘴角／上下峰及幾何插值頂點，明示不是新增語義證據。閉嘴時不補造不可見內嘴。 |
| local/semantic_mouth.py | style_profile / prepare_face | 明確 real / cartoon 分支；auto 依定位路徑作保守啟發式，可能判錯，可在 config 指定。卡通關 GFPGAN；真人已啟用修復時固定 35%。卡通邊緣銳化每通道最多約 2 灰階，不冒充超解析度。 |
| local/contour_mask.py | masks / ContourTracker | 原始與當格唇線包絡，3–5 個「輸出像素」外硬歸零，並裁在已確認嘴區內。最多保留一組前格輪廓；連續逾 5 格找不到當格輪廓即停止，沒有 5 點粗框備援。 |
| local/contour_preflight.py | prepare_source / main | 無權重的單張遮罩預覽，輸出 mouth-overlay.png、mouth-alpha.png、mouth-contour.json。 |
| local/face_worker.py | main 備援分支 | 刪除直接採用 yunet5_estimated 的結果路徑，加入熱圖品質與輪廓備援。已有眼框優先保留。 |
| local/stage_worker.py | encode 分支 | 編碼前產生輪廓預覽與風格設定，卡通輸入使用限幅邊緣處理。 |
| local/renderer.py | main | 在載入修復模型前套用風格分支；同 run_id 才讀輪廓暫存；動態唇線參與每格融合。 |
| local/quality.py | blend_mouth | 增加 mouth_geometry / feather_px；唇內、唇部、外側肌膚環分層；色彩轉移只作用外側窄環。嘴區以外像素在編碼前不變。 |
| face_detection.go | detectFace | 僅接線：重新定位已有專案時傳遞既有眼框。沒有編譯。 |
| local.go / managed_engines.go | scripts lists | 僅接線：更新 Python 子模組時複製三個新檔。沒有編譯。 |

## 明確限制

- Canny 只偵測邊緣。不是訓練好的卡通語義定位模型，無法承諾所有 2D／3D 畫風成功。
- 3–5 px 是量測唇線外的硬限制，另以嘴區作安全邊界；若量測本身錯誤，仍須在疊圖審核時拒絕，不能把像素限制當成人體語義保證。
- 預覽紅色是重繪與外側羽化；黃線是量測唇線。請確認沒有選到鼻子、臉頰或下巴。mouth-alpha.png 為臉框內的單通道遮罩，不是全圖尺寸。
- 關閉卡通寫實修復可避免該修復器重畫皮膚／牙齒，但原嘴型模型仍可能生成不同牙形；沒有硬貼原始牙齒，也沒有宣稱逐格牙形已完全保真。
- 目前以既有勾選決定是否載入真人 GFPGAN；已啟用時採 35%，未安裝／未啟用不強迫下載。完整一鍵安裝與 UI 風格選單留待本階段確認後接線。
- 眨眼、視線、眉毛及身體動作函式與 v0.8.2 一致。新嘴部路徑不修改 blink／gaze／eye_boxes；只有重新做五官定位且沒有已保存眼框時才需要新的眼部輪廓。
- 主路徑為 MuseTalk 分階段模式；尚未為外部整臉／姿勢引擎提供同等驗證。舊的非唇線呼叫保留相容性，不應誤認為也具備本階段的幾何限制。
- 原有程序隔離、4～6 GiB 動態預算、CPU 回退未更動。新模組僅用已存在的 NumPy／OpenCV，沒有額外神經權重。

## 靜態預覽用法（本機既有 Python 子模組）

```powershell
# 使用已安裝程式的獨立 Python，不須重裝系統 Python
& "$env:APPDATA\JunbaTalkVideo\local-v0.2\python\python.exe" -I local/contour_preflight.py --config preview-config.json --out mask-preview
```

preview-config.json 需包含 source（照片絕對路徑）、bbox（完整臉部 x/y/w/h，0～1）、mouth_box（嘴部搜尋安全範圍）、style_type（auto/real/cartoon）、contour_feather_px（3/4/5）。可附 98 點 landmarks 或 mouth_geometry。先用既有專案 local-config.json 搭配 style_type=cartoon 亦可；勿將舊的 5 點視為精細唇線。

第一階段驗收：靜態唇線疊圖正確 → Windows 本機 1 秒語音短片 → 檢查牙齒／外圈接縫／眨眼仍自然 → 再決定是否開放長片。本包未編譯新 EXE，未跑真實權重、GPU/NPU 或音畫同步驗收。


## 本機驗證與比對（不編譯、不生成長片）

1. 解壓核心 Source ZIP，雙擊 `Verify_Mask.cmd`。
2. 選現有專案的 `local-config.json` 可沿用已確認的臉／嘴／眼框與 98 點；或選新照片，工具會使用既有五官模型，在獨立子程序定位，結束後才合成預覽。沒有既有模型時不會下載、也不猜粗嘴框。
3. 比對圖左為原圖、中為紅色重繪範圍與黃色唇線、右為既存嘴型快取融合；沒有快取時右側仍為原圖，報告明示沒有驗證生成後牙齒。
4. `verification.json` 會列出 3–5 px 外是否硬歸零；有快取時也會核對重繪區外及眼框像素是否未改變。
5. 靜態圖不能證明動態牙齒保形、對嘴與眨眼自然；確認遮罩後，再於本機另做 1 秒短片驗收。這個工具不觸發神經嘴型生成、不編譯或打包 EXE。

命令列可重現設定：

```powershell
& "$env:APPDATA\JunbaTalkVideo\local-v0.2\python\python.exe" -I verify_mask.py --project "C:\您的專案資料夾" --style cartoon --feather 4 --out "C:\您的遮罩比對資料夾"
```

指定 `--cached-frame "C:\專案\mouth-cache\00100.png"` 可比對特定已生成嘴型格；省略時，若找到 mouth-cache，會取中間一格。可以重跑 `--feather 3` 或 `--feather 5` 做靜態比較。重跑會建立新的報告資料夾，不改原圖、原影片或現有專案。

修訂：舊的粗搜尋框可能截斷上揚嘴角，邊緣搜尋先向左右各擴 6%、上下各擴 24%，以 3×3 閉運算連接短邊緣缺口。成品遮罩仍只採量測唇線，安全範圍由量測輪廓加指定 3–5 px 得到；原 seed 與新範圍均寫入報告。搜尋框擴展不等於重繪框擴展。

修訂：笑容的口腔暗區不能當成完整外唇。備援加入 Lab 色彩對比候選（牙齒低紅度與深色口腔），並在二值幾何候選上移除細笑紋支線；不磨平來源照片。這是啟發式量測，仍須看疊圖，不是已訓練好的卡通語義網格。

安全補充：原始與當格嘴型以多邊形精確聯集處理，不用凸包跨過笑容的凹形上緣，以免重繪鼻下皮膚。GitHub 工作流程須手動觸發並勾選 mask_review_approved=true；push／PR 不會編譯 EXE。本輪沒有執行工作流程。
